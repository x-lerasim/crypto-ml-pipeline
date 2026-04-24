package cryptoml.ingestion

import org.slf4j.LoggerFactory
import play.api.libs.json._
import sttp.client3._

import scala.concurrent.duration._
import scala.util.{Failure, Success, Try}

final case class CoinCapPoint(
    priceUsd: Double,
    time:     Long,
    date:     String
)

object CoinCapPoint {
  implicit val reads: Reads[CoinCapPoint] = Json.reads[CoinCapPoint]
}

final case class CoinCapConfig(
    baseUrl:     String,
    asset:       String,
    interval:    String,
    historyDays: Int,
    timeoutMs:   Long
)

final class CoinCapClient(cfg: CoinCapConfig) {

  private val log     = LoggerFactory.getLogger(getClass)
  private val backend = HttpClientSyncBackend(
    options = SttpBackendOptions.connectionTimeout(cfg.timeoutMs.millis)
  )

  def close(): Unit = backend.close()

  def fetchHistory(): Seq[CoinCapPoint] = {
    val now   = System.currentTimeMillis()
    val start = now - cfg.historyDays.toLong * 24L * 3600L * 1000L
    fetchRange(start, now)
  }

  def fetchRange(startMs: Long, endMs: Long): Seq[CoinCapPoint] = {
    val url = uri"${cfg.baseUrl}/assets/${cfg.asset}/history?interval=${cfg.interval}&start=$startMs&end=$endMs"
    log.info(s"GET $url")

    val response = basicRequest
      .get(url)
      .header("Accept", "application/json")
      .readTimeout(cfg.timeoutMs.millis)
      .send(backend)

    response.body match {
      case Right(body) => parse(body)
      case Left(err)   =>
        log.error(s"CoinCap request failed: $err")
        throw new RuntimeException(s"CoinCap error: $err")
    }
  }

  private def parse(body: String): Seq[CoinCapPoint] = {
    Try(Json.parse(body)) match {
      case Failure(ex) =>
        throw new RuntimeException(s"Invalid JSON from CoinCap: ${ex.getMessage}")
      case Success(json) =>
        (json \ "data").validate[JsArray] match {
          case JsError(errs) =>
            throw new RuntimeException(s"Unexpected payload: $errs")
          case JsSuccess(arr, _) =>
            arr.value.iterator.flatMap(parsePoint).toVector
        }
    }
  }

  private def parsePoint(v: JsValue): Option[CoinCapPoint] = {
    val priceOpt = (v \ "priceUsd").asOpt[String].flatMap(s => Try(s.toDouble).toOption)
    val timeOpt  = (v \ "time").asOpt[Long]
    val dateOpt  = (v \ "date").asOpt[String]
    for {
      p <- priceOpt
      t <- timeOpt
      d <- dateOpt
    } yield CoinCapPoint(p, t, d)
  }
}

object CoinCapClient {
  def apply(cfg: CoinCapConfig): CoinCapClient = new CoinCapClient(cfg)
}
