package cryptoml.evaluation

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.DataFrame
import org.slf4j.LoggerFactory
import play.api.libs.json.{JsObject, Json}

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

final case class Metrics(name: String, rmse: Double, mae: Double, r2: Double) {
  def toJson: JsObject = Json.obj(
    "model" -> name,
    "rmse"  -> rmse,
    "mae"   -> mae,
    "r2"    -> r2
  )

  def pretty: String =
    f"$name%-18s  RMSE=$rmse%10.4f  MAE=$mae%10.4f  R2=$r2%7.4f"
}

object ModelEvaluator {

  private val log = LoggerFactory.getLogger(getClass)

  def evaluate(name: String, model: PipelineModel, test: DataFrame, labelCol: String): Metrics = {
    val preds = model.transform(test)
    val base = new RegressionEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol("prediction")

    val rmse = base.setMetricName("rmse").evaluate(preds)
    val mae  = base.setMetricName("mae").evaluate(preds)
    val r2   = base.setMetricName("r2").evaluate(preds)

    Metrics(name, rmse, mae, r2)
  }

  def report(metrics: Seq[Metrics]): Unit = {
    val bar = "-" * 60
    log.info(bar)
    log.info("Model evaluation on held-out tail:")
    metrics.foreach(m => log.info(m.pretty))
    log.info(bar)
  }

  def saveJson(metrics: Seq[Metrics], path: String): Unit = {
    val payload = Json.obj("metrics" -> metrics.map(_.toJson))
    val p = Paths.get(path)
    Option(p.getParent).foreach(Files.createDirectories(_))
    Files.write(p, Json.prettyPrint(payload).getBytes(StandardCharsets.UTF_8))
    log.info(s"Metrics written to $path")
  }
}
