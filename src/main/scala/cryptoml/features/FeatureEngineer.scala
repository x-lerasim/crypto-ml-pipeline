package cryptoml.features

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame}
import org.slf4j.LoggerFactory

final case class FeatureConfig(
    smaWindows:       Seq[Int],
    rsiPeriod:        Int,
    lagPeriods:       Seq[Int],
    volatilityWindow: Int
)

object FeatureEngineer {

  private val log = LoggerFactory.getLogger(getClass)

  val PriceCol: String = "price"
  val LabelCol: String = "label"

  private def orderedWindow = Window.orderBy("ts")

  def build(df: DataFrame, cfg: FeatureConfig): DataFrame = {
    val w1 = orderedWindow

    val withPct = df.withColumn(
      "pct_change",
      (col(PriceCol) - lag(PriceCol, 1).over(w1)) / lag(PriceCol, 1).over(w1)
    )

    val withSma = cfg.smaWindows.foldLeft(withPct) { (acc, n) =>
      acc.withColumn(s"sma_$n", avg(col(PriceCol)).over(rollingWindow(n)))
    }

    val withLags = cfg.lagPeriods.foldLeft(withSma) { (acc, k) =>
      acc.withColumn(s"lag_$k", lag(PriceCol, k).over(w1))
    }

    val withVol = withLags.withColumn(
      s"vol_${cfg.volatilityWindow}",
      stddev(col("pct_change")).over(rollingWindow(cfg.volatilityWindow))
    )

    val withRsi = addRsi(withVol, cfg.rsiPeriod)

    val labeled = withRsi.withColumn(LabelCol, lead(PriceCol, 1).over(w1))

    val featureCols = assembledFeatureCols(cfg)
    val nonNull = labeled
      .na.drop(subset = LabelCol +: featureCols)

    log.info(s"FeatureEngineer: ${nonNull.count()} rows after feature build.")
    nonNull
  }

  def assembledFeatureCols(cfg: FeatureConfig): Seq[String] = {
    val sma  = cfg.smaWindows.map(n => s"sma_$n")
    val lags = cfg.lagPeriods.map(k => s"lag_$k")
    val rsi  = Seq(s"rsi_${cfg.rsiPeriod}")
    val vol  = Seq(s"vol_${cfg.volatilityWindow}")
    Seq("pct_change") ++ sma ++ lags ++ rsi ++ vol
  }

  private def rollingWindow(n: Int) =
    orderedWindow.rowsBetween(-(n - 1).toLong, 0L)

  private def addRsi(df: DataFrame, period: Int): DataFrame = {
    val w       = orderedWindow
    val rollW   = w.rowsBetween(-(period - 1).toLong, 0L)
    val delta   = col(PriceCol) - lag(PriceCol, 1).over(w)
    val gainCol = when(delta > 0, delta).otherwise(lit(0.0))
    val lossCol = when(delta < 0, -delta).otherwise(lit(0.0))

    val withGL = df
      .withColumn("_gain", gainCol)
      .withColumn("_loss", lossCol)

    val avgGain: Column = avg(col("_gain")).over(rollW)
    val avgLoss: Column = avg(col("_loss")).over(rollW)

    val rs  = avgGain / when(avgLoss === 0.0, lit(1e-10)).otherwise(avgLoss)
    val rsi = lit(100.0) - (lit(100.0) / (lit(1.0) + rs))

    withGL
      .withColumn(s"rsi_$period", rsi)
      .drop("_gain", "_loss")
  }
}
