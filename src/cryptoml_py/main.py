import argparse
from pyspark.sql import SparkSession
from cryptoml_py.config import load_config
from cryptoml_py.ingestion import fetch_history
from cryptoml_py.transformation import clean
from cryptoml_py.features import build, assembled_feature_cols
from cryptoml_py.model import train
from cryptoml_py.evaluation import evaluate, report, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def build_spark(cfg):
    return (
        SparkSession.builder.appName(cfg["spark"]["app-name"])
        .master(cfg["spark"]["master"])
        .config("spark.sql.shuffle.partitions", int(cfg["spark"]["shuffle-partitions"]))
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def run():
    args = parse_args()
    cfg = load_config(args.config).get("cryptoml", {})
    spark = build_spark(cfg)
    try:
        points = fetch_history(cfg["api"])
        if not points:
            raise RuntimeError("CoinCap returned no points.")
        raw = spark.createDataFrame(points)
        cleaned = clean(raw)
        feats = build(cleaned, cfg["features"])
        feature_cols = assembled_feature_cols(cfg["features"])
        feats.write.mode("overwrite").parquet(cfg["output"]["features-path"])
        trained = train(feats, cfg["model"], feature_cols)
        lr_metrics = evaluate(
            "LinearRegression",
            trained["lr_model"],
            trained["test"],
            cfg["model"]["label-col"],
        )
        gbt_metrics = evaluate(
            "GBTRegressor",
            trained["gbt_model"],
            trained["test"],
            cfg["model"]["label-col"],
        )
        all_metrics = [lr_metrics, gbt_metrics]
        report(all_metrics)
        save_json(all_metrics, cfg["output"]["metrics-path"])
    finally:
        spark.stop()


if __name__ == "__main__":
    run()
