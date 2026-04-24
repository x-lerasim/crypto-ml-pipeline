# crypto-ml-pipeline — Spark MLlib pipeline на Scala для прогнозирования цен криптовалют

End-to-end учебный ML-пайплайн: выгружает исторические свечи с публичного REST API
CoinCap, чистит и обогащает временной ряд в Apache Spark, генерирует технические
индикаторы, обучает две регрессионные модели из Spark MLlib, сравнивает их по метрикам
качества на хронологическом hold-out-е и сохраняет отчёт.

Проект создан как портфолио-демо владения **Scala + Apache Spark + Spark MLlib**.

---

## Что делает проект

1. **Ingestion** — `CoinCapClient` синхронно ходит в `GET /v2/assets/{id}/history`
   через `sttp-client3` + `play-json` и возвращает `Seq[CoinCapPoint]`.
2. **Transformation** — `DataCleaner` приводит типы, дедуплицирует по `ts` через
   `row_number()` window-функцию, отбрасывает нулевые/отрицательные цены,
   forward-fill-ит оставшиеся пропуски (`last(_, ignoreNulls=true)` по отсортированному окну).
3. **Feature engineering** — `FeatureEngineer` считает SMA, RSI, лаги, процентное
   изменение и скользящую волатильность через оконные функции Spark SQL.
4. **Training** — `PricePredictor` строит два `Pipeline` (VectorAssembler → estimator)
   и подбирает гиперпараметры `CrossValidator`-ом с k-fold внутри training-префикса.
5. **Evaluation** — `ModelEvaluator` считает RMSE / MAE / R², выводит в консоль и
   пишет `data/metrics.json`.

## Архитектура

```
  CoinCap API
      │  (HTTPS, REST)
      ▼
 ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
 │ CoinCapClient│ ───▶ │ DataCleaner  │ ───▶ │ FeatureEngineer  │
 │  (sttp)      │      │ (Spark SQL)  │      │  (window funcs)  │
 └──────────────┘      └──────────────┘      └──────────────────┘
                                                       │
                                                       ▼
                                         ┌───────────────────────────┐
                                         │ PricePredictor            │
                                         │  · LinearRegression + CV  │
                                         │  · GBTRegressor     + CV  │
                                         └───────────────────────────┘
                                                       │
                                                       ▼
                                         ┌───────────────────────────┐
                                         │ ModelEvaluator            │
                                         │  RMSE / MAE / R²  + JSON  │
                                         └───────────────────────────┘
```

## Стек

- **Scala 2.12** (совместимость с Spark)
- **Apache Spark 3.5** (spark-core, spark-sql)
- **Spark MLlib 3.5** (Pipeline, CrossValidator, LinearRegression, GBTRegressor, RegressionEvaluator)
- **sttp-client3** — HTTP к CoinCap API
- **Typesafe Config** — `application.conf`
- **play-json** — парсинг ответа API и сериализация метрик
- **ScalaTest 3.2** — юнит-тесты на локальной `SparkSession`
- **Docker** — multi-stage (sbt-assembly build → Spark runtime)

## Feature engineering

| Feature            | Описание                                                                 |
|--------------------|--------------------------------------------------------------------------|
| `sma_7`, `sma_30`  | Простая скользящая средняя цены за 7 и 30 периодов                       |
| `rsi_14`           | Relative Strength Index за 14 периодов: `100 - 100 / (1 + avgGain/avgLoss)` |
| `lag_1` … `lag_7`  | Цена закрытия с задержкой 1…7 периодов                                   |
| `pct_change`       | `(price_t - price_{t-1}) / price_{t-1}` — относительное изменение        |
| `vol_14`           | Стандартное отклонение `pct_change` в окне 14 — proxy-волатильность      |
| `label` (target)   | `lead(price, 1)` — цена следующего периода                               |

Все строки с `null`-фичами после warm-up-периода отбрасываются.

## Модели

Обучаются **две разные по природе** модели, чтобы продемонстрировать сравнение:

- **LinearRegression** — простая, быстрая, интерпретируемая; служит baseline-ом.
  Грид по `regParam` и `elasticNetParam`.
- **GBTRegressor** — градиентный бустинг на деревьях; ловит нелинейности и
  взаимодействия фичей, но дороже. Грид по `maxDepth` и `maxIter`.

Обе обёрнуты в `Pipeline` с `VectorAssembler` и подбираются `CrossValidator`-ом
(по умолчанию 3 фолда) внутри **хронологически взятых 80% данных**. Tuner не видит
последние 20% — на них и считаются метрики.

## Пример вывода метрик

```
------------------------------------------------------------
Model evaluation on held-out tail:
LinearRegression    RMSE=   612.4321  MAE=   488.1100  R²= 0.9421
GBTRegressor        RMSE=   503.9180  MAE=   401.7500  R²= 0.9608
------------------------------------------------------------
Metrics written to data/metrics.json
```

И содержимое `data/metrics.json`:

```json
{
  "metrics": [
    { "model": "LinearRegression", "rmse": 612.4321, "mae": 488.1100, "r2": 0.9421 },
    { "model": "GBTRegressor",     "rmse": 503.9180, "mae": 401.7500, "r2": 0.9608 }
  ]
}
```

## Запуск

### Локально (sbt)

Требования: JDK 11, sbt 1.9.x.

```bash
sbt test        # юнит-тесты на локальном SparkSession
sbt run         # полный пайплайн: ingestion → features → train → evaluate
```

По умолчанию берётся `bitcoin`, интервал `d1`, последние 365 дней. Поменять
можно через `src/main/resources/application.conf`.

### Docker

```bash
docker build -t crypto-ml-pipeline .
docker run --rm -v "$PWD/data:/opt/cryptoml/data" crypto-ml-pipeline
```

Multi-stage сборка: первая стадия собирает fat jar через `sbt-assembly` с
`-DsparkScope=provided`, вторая — запускает его на официальном образе
`apache/spark:3.5.1`.

## Конфигурация (`application.conf`)

- `api.asset` — CoinCap asset id (`bitcoin`, `ethereum`, `solana`, …)
- `api.interval` — `m1 | m5 | m15 | m30 | h1 | h2 | h6 | h12 | d1`
- `api.history-days` — глубина истории
- `features.*` — окна SMA, период RSI, лаги, окно волатильности
- `model.*` — доля train, число фолдов CV, гиперпараметры для обоих грид-сёрчей
- `output.metrics-path` — куда писать итоговый JSON

## Структура проекта

```
crypto-ml-pipeline/
├── build.sbt
├── project/
│   ├── build.properties            # sbt 1.9.9
│   └── plugins.sbt                 # sbt-assembly
├── src/main/scala/cryptoml/
│   ├── ingestion/CoinCapClient.scala
│   ├── transformation/DataCleaner.scala
│   ├── features/FeatureEngineer.scala
│   ├── model/PricePredictor.scala
│   ├── evaluation/ModelEvaluator.scala
│   └── Main.scala
├── src/main/resources/
│   ├── application.conf
│   └── logback.xml
├── src/test/scala/cryptoml/
│   ├── SparkTestBase.scala
│   ├── DataCleanerSpec.scala
│   ├── FeatureEngineerSpec.scala
│   └── ModelEvaluatorSpec.scala
├── data/                          # генерируемые parquet/json (в .gitignore)
├── Dockerfile
├── .gitignore
└── README.md
```
