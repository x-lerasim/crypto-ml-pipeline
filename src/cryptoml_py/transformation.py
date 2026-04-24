from pyspark.sql import functions as F
from pyspark.sql import Window


def forward_fill(df, target, order_col):
    w = Window.orderBy(order_col).rowsBetween(Window.unboundedPreceding, Window.currentRow)
    return df.withColumn(target, F.last(F.col(target), ignorenulls=True).over(w))


def clean(raw_df):
    typed = (
        raw_df.select(
            ((F.col("time") / F.lit(1000)).cast("timestamp")).alias("ts"),
            F.col("priceUsd").cast("double").alias("price"),
        )
        .filter(F.col("price").isNotNull() & (F.col("price") > F.lit(0.0)))
        .withColumn("date", F.to_date(F.col("ts")))
    )
    dedup_window = Window.partitionBy("ts").orderBy(F.col("price").desc())
    deduped = (
        typed.withColumn("_rn", F.row_number().over(dedup_window))
        .filter(F.col("_rn") == F.lit(1))
        .drop("_rn")
        .orderBy("ts")
    )
    filled = forward_fill(deduped, "price", "ts")
    return filled.select("ts", "date", "price")
