import json
from pathlib import Path
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from cryptoml_py.evaluation import evaluate, save_json


def test_evaluate_perfect_tiny_model(spark):
    data = spark.createDataFrame([(float(x), 2.0 * x + 1.0) for x in range(1, 51)], ["x", "label"])
    assembler = VectorAssembler(inputCols=["x"], outputCol="features")
    lr = LinearRegression(featuresCol="features", labelCol="label")
    pipeline = Pipeline(stages=[assembler, lr]).fit(data)
    m = evaluate("tiny-lr", pipeline, data, "label")
    assert m["model"] == "tiny-lr"
    assert m["rmse"] < 1e-6
    assert m["mae"] < 1e-6
    assert abs(m["r2"] - 1.0) <= 1e-6


def test_save_json_valid(tmp_path: Path):
    target = tmp_path / "metrics.json"
    metrics = [
        {"model": "A", "rmse": 1.0, "mae": 0.5, "r2": 0.9},
        {"model": "B", "rmse": 2.0, "mae": 1.0, "r2": 0.8},
    ]
    save_json(metrics, str(target))
    body = json.loads(target.read_text(encoding="utf-8"))
    assert len(body["metrics"]) == 2
    assert body["metrics"][0]["model"] == "A"
