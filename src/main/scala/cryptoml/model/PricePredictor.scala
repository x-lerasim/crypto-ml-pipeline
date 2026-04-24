package cryptoml.model

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.slf4j.LoggerFactory

final case class LrHyperGrid(regParam: Seq[Double], elasticNet: Seq[Double])
final case class GbtHyperGrid(maxDepth: Seq[Int], maxIter: Seq[Int])

final case class ModelConfig(
    featureCols:   Seq[String],
    labelCol:      String,
    trainFraction: Double,
    cvFolds:       Int,
    lr:            LrHyperGrid,
    gbt:           GbtHyperGrid
)

final case class TrainingResult(
    lrModel:  PipelineModel,
    gbtModel: PipelineModel,
    train:    DataFrame,
    test:     DataFrame
)

object PricePredictor {

  private val log = LoggerFactory.getLogger(getClass)

  def train(features: DataFrame, cfg: ModelConfig): TrainingResult = {
    val (train, test) = chronologicalSplit(features, cfg.trainFraction)
    log.info(s"Train rows: ${train.count()}, Test rows: ${test.count()}")

    val assembler = new VectorAssembler()
      .setInputCols(cfg.featureCols.toArray)
      .setOutputCol("features")
      .setHandleInvalid("skip")

    val evaluator = new RegressionEvaluator()
      .setLabelCol(cfg.labelCol)
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val lrModel  = trainLinear(train, assembler, evaluator, cfg)
    val gbtModel = trainGbt(train,    assembler, evaluator, cfg)

    TrainingResult(lrModel, gbtModel, train, test)
  }

  private def chronologicalSplit(df: DataFrame, frac: Double): (DataFrame, DataFrame) = {
    val ordered = df.orderBy(col("ts").asc).cache()
    val total   = ordered.count()
    val cutoff  = math.max(1L, (total * frac).toLong)
    val train = ordered.limit(cutoff.toInt)
    val test  = ordered
      .withColumn("_rn", monotonically_increasing_id())
      .filter(col("_rn") >= cutoff)
      .drop("_rn")
    (train, test)
  }

  private def trainLinear(
      train:     DataFrame,
      assembler: VectorAssembler,
      evaluator: RegressionEvaluator,
      cfg:       ModelConfig
  ): PipelineModel = {
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol(cfg.labelCol)

    val pipeline = new Pipeline().setStages(Array(assembler, lr))

    val grid = new ParamGridBuilder()
      .addGrid(lr.regParam,        cfg.lr.regParam.toArray)
      .addGrid(lr.elasticNetParam, cfg.lr.elasticNet.toArray)
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(grid)
      .setNumFolds(cfg.cvFolds)
      .setParallelism(2)

    log.info("Fitting LinearRegression with CV...")
    cv.fit(train).bestModel.asInstanceOf[PipelineModel]
  }

  private def trainGbt(
      train:     DataFrame,
      assembler: VectorAssembler,
      evaluator: RegressionEvaluator,
      cfg:       ModelConfig
  ): PipelineModel = {
    val gbt = new GBTRegressor()
      .setFeaturesCol("features")
      .setLabelCol(cfg.labelCol)
      .setSeed(42L)

    val pipeline = new Pipeline().setStages(Array(assembler, gbt))

    val grid = new ParamGridBuilder()
      .addGrid(gbt.maxDepth, cfg.gbt.maxDepth.toArray)
      .addGrid(gbt.maxIter,  cfg.gbt.maxIter.toArray)
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(grid)
      .setNumFolds(cfg.cvFolds)
      .setParallelism(2)

    log.info("Fitting GBTRegressor with CV...")
    cv.fit(train).bestModel.asInstanceOf[PipelineModel]
  }
}
