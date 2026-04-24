package cryptoml

import cryptoml.features.{FeatureConfig, FeatureEngineer}
import org.apache.spark.sql.functions._

import java.sql.Timestamp

class FeatureEngineerSpec extends SparkTestBase {

  private def ts(i: Int): Timestamp = new Timestamp(1_700_000_000_000L + i.toLong * 86_400_000L)

  "FeatureEngineer.build" should "produce requested feature columns and a label" in {
    import spark.implicits._
    val prices = (0 until 40).map(i => 100.0 + i * 0.5 + math.sin(i / 3.0) * 4.0)
    val rows   = prices.zipWithIndex.map { case (p, i) =>
      (ts(i), java.sql.Date.valueOf("2023-11-14"), p)
    }
    val df = rows.toDF("ts", "date", "price")

    val cfg = FeatureConfig(
      smaWindows       = Seq(7, 14),
      rsiPeriod        = 14,
      lagPeriods       = Seq(1, 2, 3),
      volatilityWindow = 14
    )

    val out  = FeatureEngineer.build(df, cfg)
    val cols = out.columns.toSet

    Seq("sma_7", "sma_14", "rsi_14", "vol_14", "lag_1", "lag_2", "lag_3", "pct_change", "label")
      .foreach(c => withClue(s"missing column $c: ")(cols should contain (c)))

    out.count() should be > 0L

    val head = out.orderBy("ts").limit(1).collect().head
    val headTs = head.getAs[Timestamp]("ts")
    val headLabel = head.getAs[Double]("label")

    val nextPrice = df
      .filter(col("ts") > lit(headTs))
      .orderBy("ts")
      .limit(1)
      .collect()
      .head
      .getAs[Double]("price")
    headLabel shouldBe nextPrice +- 1e-9
  }

  it should "expose feature cols in a stable order" in {
    val cfg = FeatureConfig(Seq(7, 30), 14, Seq(1, 2, 3, 4, 5, 6, 7), 14)
    val cols = FeatureEngineer.assembledFeatureCols(cfg)
    cols.head shouldBe "pct_change"
    cols should contain allOf ("sma_7", "sma_30", "rsi_14", "vol_14", "lag_1", "lag_7")
  }
}
