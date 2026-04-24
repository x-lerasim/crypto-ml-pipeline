package cryptoml.transformation

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.LoggerFactory

object DataCleaner {

  private val log = LoggerFactory.getLogger(getClass)

  object Cols {
    val Ts    = "ts"
    val Date  = "date"
    val Price = "price"
  }

  def clean(raw: DataFrame): DataFrame = {
    val spark = raw.sparkSession
    import spark.implicits._

    val typed = raw
      .select(
        (col("time") / 1000L).cast(TimestampType).as(Cols.Ts),
        col("priceUsd").cast(DoubleType).as(Cols.Price)
      )
      .filter(col(Cols.Price).isNotNull && col(Cols.Price) > 0.0)
      .withColumn(Cols.Date, to_date(col(Cols.Ts)))

    val dedupWindow = Window.partitionBy(Cols.Ts).orderBy(col(Cols.Price).desc)
    val deduped = typed
      .withColumn("_rn", row_number().over(dedupWindow))
      .filter(col("_rn") === 1)
      .drop("_rn")
      .orderBy(Cols.Ts)

    val filled = forwardFill(deduped, Cols.Price, orderCol = Cols.Ts)

    log.info(s"DataCleaner: ${filled.count()} rows after cleaning")
    filled.select(Cols.Ts, Cols.Date, Cols.Price)
  }

  def forwardFill(df: DataFrame, target: String, orderCol: String): DataFrame = {
    val w = Window
      .orderBy(orderCol)
      .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    df.withColumn(target, last(col(target), ignoreNulls = true).over(w))
  }
}
