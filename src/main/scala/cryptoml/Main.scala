package cryptoml

import com.typesafe.config.{Config, ConfigFactory}
import cryptoml.evaluation.ModelEvaluator
import cryptoml.features.{FeatureConfig, FeatureEngineer}
import cryptoml.ingestion.{CoinCapClient, CoinCapConfig, CoinCapPoint}
import cryptoml.model.{GbtHyperGrid, LrHyperGrid, ModelConfig, PricePredictor}
import cryptoml.transformation.DataCleaner
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

object Main {

  private val log = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    val cfg   = ConfigFactory.load().getConfig("cryptoml")
    val spark = buildSpark(cfg)

    try {
      val apiCfg     = readApiCfg(cfg)
      val featCfg    = readFeatureCfg(cfg)
      val modelCfgFn = readModelCfgFn(cfg)

      log.info(s"Fetching ${apiCfg.historyDays} days of ${apiCfg.asset} @ ${apiCfg.interval}")
      val raw = ingest(spark, apiCfg)

      val cleaned = DataCleaner.clean(raw)
      val feats   = FeatureEngineer.build(cleaned, featCfg)
      val modelCfg = modelCfgFn(FeatureEngineer.assembledFeatureCols(featCfg))

      feats.write.mode("overwrite").parquet(cfg.getString("output.features-path"))
      log.info(s"Features persisted to ${cfg.getString("output.features-path")}")

      val trained = PricePredictor.train(feats, modelCfg)

      val lrMetrics  = ModelEvaluator.evaluate("LinearRegression", trained.lrModel,  trained.test, modelCfg.labelCol)
      val gbtMetrics = ModelEvaluator.evaluate("GBTRegressor",     trained.gbtModel, trained.test, modelCfg.labelCol)
      val all = Seq(lrMetrics, gbtMetrics)

      ModelEvaluator.report(all)
      ModelEvaluator.saveJson(all, cfg.getString("output.metrics-path"))
    } finally {
      spark.stop()
    }
  }

  private def buildSpark(cfg: Config): SparkSession =
    SparkSession.builder()
      .appName(cfg.getString("spark.app-name"))
      .master(cfg.getString("spark.master"))
      .config("spark.sql.shuffle.partitions", cfg.getInt("spark.shuffle-partitions"))
      .config("spark.ui.showConsoleProgress", "false")
      .getOrCreate()

  private def ingest(spark: SparkSession, cfg: CoinCapConfig): DataFrame = {
    val client = CoinCapClient(cfg)
    try {
      val points: Seq[CoinCapPoint] = client.fetchHistory()
      require(points.nonEmpty, "CoinCap returned no points.")
      import spark.implicits._
      points.toDF()
    } finally {
      client.close()
    }
  }

  private def readApiCfg(c: Config): CoinCapConfig = CoinCapConfig(
    baseUrl     = c.getString("api.base-url"),
    asset       = c.getString("api.asset"),
    interval    = c.getString("api.interval"),
    historyDays = c.getInt("api.history-days"),
    timeoutMs   = c.getLong("api.timeout-ms")
  )

  private def readFeatureCfg(c: Config): FeatureConfig = FeatureConfig(
    smaWindows       = c.getIntList("features.sma-windows").asScala.map(_.toInt).toSeq,
    rsiPeriod        = c.getInt("features.rsi-period"),
    lagPeriods       = c.getIntList("features.lag-periods").asScala.map(_.toInt).toSeq,
    volatilityWindow = c.getInt("features.volatility-window")
  )

  private def readModelCfgFn(c: Config): Seq[String] => ModelConfig = { featureCols =>
    ModelConfig(
      featureCols   = featureCols,
      labelCol      = c.getString("model.label-col"),
      trainFraction = c.getDouble("model.train-fraction"),
      cvFolds       = c.getInt("model.cv-folds"),
      lr  = LrHyperGrid(
        regParam   = c.getDoubleList("model.lr.reg-param").asScala.map(_.toDouble).toSeq,
        elasticNet = c.getDoubleList("model.lr.elastic-net-param").asScala.map(_.toDouble).toSeq
      ),
      gbt = GbtHyperGrid(
        maxDepth = c.getIntList("model.gbt.max-depth").asScala.map(_.toInt).toSeq,
        maxIter  = c.getIntList("model.gbt.max-iter").asScala.map(_.toInt).toSeq
      )
    )
  }
}
