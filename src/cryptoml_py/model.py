from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
from pyspark.sql import Window


def chronological_split(df, frac):
    ordered = df.orderBy(F.col("ts").asc()).cache()
    with_rn = ordered.withColumn("_rn", F.row_number().over(Window.orderBy(F.col("ts").asc())))
    total = with_rn.count()
    cutoff = max(1, int(total * float(frac)))
    train = with_rn.filter(F.col("_rn") <= F.lit(cutoff)).drop("_rn")
    test = with_rn.filter(F.col("_rn") > F.lit(cutoff)).drop("_rn")
    return train, test


def train_linear(train_df, assembler, evaluator, model_cfg):
    lr = LinearRegression().setFeaturesCol("features").setLabelCol(model_cfg["label-col"])
    pipeline = Pipeline(stages=[assembler, lr])
    grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [float(x) for x in model_cfg["lr"]["reg-param"]])
        .addGrid(lr.elasticNetParam, [float(x) for x in model_cfg["lr"]["elastic-net-param"]])
        .build()
    )
    cv = (
        CrossValidator()
        .setEstimator(pipeline)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(grid)
        .setNumFolds(int(model_cfg["cv-folds"]))
        .setParallelism(2)
    )
    return cv.fit(train_df).bestModel


def train_gbt(train_df, assembler, evaluator, model_cfg):
    gbt = (
        GBTRegressor()
        .setFeaturesCol("features")
        .setLabelCol(model_cfg["label-col"])
        .setSeed(42)
    )
    pipeline = Pipeline(stages=[assembler, gbt])
    grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [int(x) for x in model_cfg["gbt"]["max-depth"]])
        .addGrid(gbt.maxIter, [int(x) for x in model_cfg["gbt"]["max-iter"]])
        .build()
    )
    cv = (
        CrossValidator()
        .setEstimator(pipeline)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(grid)
        .setNumFolds(int(model_cfg["cv-folds"]))
        .setParallelism(2)
    )
    return cv.fit(train_df).bestModel


def train(features_df, model_cfg, feature_cols):
    train_df, test_df = chronological_split(features_df, model_cfg["train-fraction"])
    assembler = (
        VectorAssembler()
        .setInputCols(feature_cols)
        .setOutputCol("features")
        .setHandleInvalid("skip")
    )
    evaluator = (
        RegressionEvaluator()
        .setLabelCol(model_cfg["label-col"])
        .setPredictionCol("prediction")
        .setMetricName("rmse")
    )
    lr_model = train_linear(train_df, assembler, evaluator, model_cfg)
    gbt_model = train_gbt(train_df, assembler, evaluator, model_cfg)
    return {
        "lr_model": lr_model,
        "gbt_model": gbt_model,
        "train": train_df,
        "test": test_df,
    }
