from pyspark.sql import functions as F
from pyspark.sql import Window


def rolling_window(n):
    return Window.orderBy("ts").rowsBetween(-(n - 1), 0)


def add_rsi(df, period):
    w = Window.orderBy("ts")
    roll_w = w.rowsBetween(-(period - 1), 0)
    delta = F.col("price") - F.lag("price", 1).over(w)
    gain_col = F.when(delta > F.lit(0), delta).otherwise(F.lit(0.0))
    loss_col = F.when(delta < F.lit(0), -delta).otherwise(F.lit(0.0))
    with_gl = df.withColumn("_gain", gain_col).withColumn("_loss", loss_col)
    avg_gain = F.avg(F.col("_gain")).over(roll_w)
    avg_loss = F.avg(F.col("_loss")).over(roll_w)
    rs = avg_gain / F.when(avg_loss == F.lit(0.0), F.lit(1e-10)).otherwise(avg_loss)
    rsi = F.lit(100.0) - (F.lit(100.0) / (F.lit(1.0) + rs))
    return with_gl.withColumn(f"rsi_{period}", rsi).drop("_gain", "_loss")


def assembled_feature_cols(feature_cfg):
    sma = [f"sma_{n}" for n in feature_cfg["sma-windows"]]
    lags = [f"lag_{k}" for k in feature_cfg["lag-periods"]]
    rsi = [f'rsi_{feature_cfg["rsi-period"]}']
    vol = [f'vol_{feature_cfg["volatility-window"]}']
    return ["pct_change"] + sma + lags + rsi + vol


def build(df, feature_cfg):
    w1 = Window.orderBy("ts")
    with_pct = df.withColumn(
        "pct_change",
        (F.col("price") - F.lag("price", 1).over(w1)) / F.lag("price", 1).over(w1),
    )
    with_sma = with_pct
    for n in feature_cfg["sma-windows"]:
        with_sma = with_sma.withColumn(f"sma_{n}", F.avg(F.col("price")).over(rolling_window(n)))
    with_lags = with_sma
    for k in feature_cfg["lag-periods"]:
        with_lags = with_lags.withColumn(f"lag_{k}", F.lag("price", k).over(w1))
    with_vol = with_lags.withColumn(
        f'vol_{feature_cfg["volatility-window"]}',
        F.stddev(F.col("pct_change")).over(rolling_window(feature_cfg["volatility-window"])),
    )
    with_rsi = add_rsi(with_vol, int(feature_cfg["rsi-period"]))
    labeled = with_rsi.withColumn("label", F.lead("price", 1).over(w1))
    feature_cols = assembled_feature_cols(feature_cfg)
    return labeled.na.drop(subset=["label"] + feature_cols)
