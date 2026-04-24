package cryptoml

import cryptoml.transformation.DataCleaner
import org.apache.spark.sql.types._

class DataCleanerSpec extends SparkTestBase {

  "DataCleaner.clean" should "cast, dedupe, drop non-positive prices and produce expected schema" in {
    import spark.implicits._

    val raw = Seq(
      (1_700_000_000_000L, "100.0"),
      (1_700_086_400_000L, "110.0"),
      (1_700_086_400_000L, "108.0"),
      (1_700_172_800_000L, "0.0"),
      (1_700_259_200_000L, "120.0")
    ).toDF("time", "priceUsd")

    val cleaned = DataCleaner.clean(raw)

    cleaned.schema.fieldNames.toSet shouldBe Set("ts", "date", "price")
    cleaned.schema("price").dataType shouldBe DoubleType
    cleaned.schema("ts").dataType    shouldBe TimestampType

    val rows = cleaned.orderBy("ts").collect().map(r => r.getAs[Double]("price")).toSeq
    rows shouldBe Seq(100.0, 110.0, 120.0)
  }

  it should "forward-fill nulls on the target column" in {
    import spark.implicits._
    val df = Seq(
      (1L, Option(1.0)),
      (2L, Option.empty[Double]),
      (3L, Option(3.0)),
      (4L, Option.empty[Double])
    ).toDF("ts", "price")

    val filled = DataCleaner.forwardFill(df, "price", "ts")
    val vals = filled.orderBy("ts").collect().map(_.getAs[Double]("price")).toSeq
    vals shouldBe Seq(1.0, 1.0, 3.0, 3.0)
  }
}
