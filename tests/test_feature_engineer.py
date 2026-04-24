import datetime
from pyspark.sql import functions as F
from cryptoml_py.features import build, assembled_feature_cols


def ts(i):
    return datetime.datetime.fromtimestamp((1700000000000 + i * 86400000) / 1000.0)


def test_build_features_and_label(spark):
    prices = [100.0 + i * 0.5 + __import__("math").sin(i / 3.0) * 4.0 for i in range(40)]
    rows = [(ts(i), datetime.date(2023, 11, 14), p) for i, p in enumerate(prices)]
    df = spark.createDataFrame(rows, ["ts", "date", "price"])
    cfg = {
        "sma-windows": [7, 14],
        "rsi-period": 14,
        "lag-periods": [1, 2, 3],
        "volatility-window": 14,
    }
    out = build(df, cfg)
    cols = set(out.columns)
    for col in ["sma_7", "sma_14", "rsi_14", "vol_14", "lag_1", "lag_2", "lag_3", "pct_change", "label"]:
        assert col in cols
    assert out.count() > 0
    head = out.orderBy("ts").limit(1).collect()[0]
    head_ts = head["ts"]
    head_label = head["label"]
    next_price = (
        df.filter(F.col("ts") > F.lit(head_ts))
        .orderBy("ts")
        .limit(1)
        .collect()[0]["price"]
    )
    assert abs(head_label - next_price) <= 1e-9


def test_feature_cols_order():
    cfg = {
        "sma-windows": [7, 30],
        "rsi-period": 14,
        "lag-periods": [1, 2, 3, 4, 5, 6, 7],
        "volatility-window": 14,
    }
    cols = assembled_feature_cols(cfg)
    assert cols[0] == "pct_change"
    for col in ["sma_7", "sma_30", "rsi_14", "vol_14", "lag_1", "lag_7"]:
        assert col in cols
