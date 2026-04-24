from pyspark.sql.types import DoubleType, TimestampType
from cryptoml_py.transformation import clean, forward_fill


def test_clean_cast_dedupe_drop_and_schema(spark):
    raw = spark.createDataFrame(
        [
            (1700000000000, "100.0"),
            (1700086400000, "110.0"),
            (1700086400000, "108.0"),
            (1700172800000, "0.0"),
            (1700259200000, "120.0"),
        ],
        ["time", "priceUsd"],
    )
    cleaned = clean(raw)
    assert set(cleaned.schema.fieldNames()) == {"ts", "date", "price"}
    assert isinstance(cleaned.schema["price"].dataType, DoubleType)
    assert isinstance(cleaned.schema["ts"].dataType, TimestampType)
    rows = [r["price"] for r in cleaned.orderBy("ts").collect()]
    assert rows == [100.0, 110.0, 120.0]


def test_forward_fill_nulls(spark):
    df = spark.createDataFrame(
        [
            (1, 1.0),
            (2, None),
            (3, 3.0),
            (4, None),
        ],
        ["ts", "price"],
    )
    filled = forward_fill(df, "price", "ts")
    vals = [r["price"] for r in filled.orderBy("ts").collect()]
    assert vals == [1.0, 1.0, 3.0, 3.0]
