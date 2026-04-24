package cryptoml

import cryptoml.evaluation.ModelEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import play.api.libs.json.Json

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

class ModelEvaluatorSpec extends SparkTestBase {

  "ModelEvaluator.evaluate" should "compute finite RMSE/MAE/R2 against a perfectly-fit tiny model" in {
    import spark.implicits._
    val data = (1 to 50).map(x => (x.toDouble, 2.0 * x + 1.0)).toDF("x", "label")

    val assembler = new VectorAssembler().setInputCols(Array("x")).setOutputCol("features")
    val lr        = new LinearRegression().setFeaturesCol("features").setLabelCol("label")
    val pipeline  = new Pipeline().setStages(Array(assembler, lr)).fit(data)

    val m = ModelEvaluator.evaluate("tiny-lr", pipeline, data, "label")

    m.name shouldBe "tiny-lr"
    m.rmse should be < 1e-6
    m.mae  should be < 1e-6
    m.r2   shouldBe 1.0 +- 1e-6
  }

  it should "serialize metrics to a valid JSON file" in {
    val tmp = Files.createTempFile("metrics-", ".json")
    try {
      val ms = Seq(
        cryptoml.evaluation.Metrics("A", 1.0, 0.5, 0.9),
        cryptoml.evaluation.Metrics("B", 2.0, 1.0, 0.8)
      )
      ModelEvaluator.saveJson(ms, tmp.toString)

      val body = new String(Files.readAllBytes(tmp), StandardCharsets.UTF_8)
      val js   = Json.parse(body)
      (js \ "metrics").as[Seq[play.api.libs.json.JsValue]] should have size 2
      ((js \ "metrics")(0) \ "model").as[String] shouldBe "A"
    } finally Files.deleteIfExists(tmp)
  }
}
