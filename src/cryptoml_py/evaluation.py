import json
from pathlib import Path
from pyspark.ml.evaluation import RegressionEvaluator


def evaluate(name, model, test_df, label_col):
    preds = model.transform(test_df)
    evaluator = RegressionEvaluator().setLabelCol(label_col).setPredictionCol("prediction")
    rmse = evaluator.setMetricName("rmse").evaluate(preds)
    mae = evaluator.setMetricName("mae").evaluate(preds)
    r2 = evaluator.setMetricName("r2").evaluate(preds)
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def report(metrics):
    for item in metrics:
        print(
            f'{item["model"]:<18}  RMSE={item["rmse"]:>10.4f}  '
            f'MAE={item["mae"]:>10.4f}  R2={item["r2"]:>7.4f}'
        )


def save_json(metrics, path):
    out = {"metrics": metrics}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
