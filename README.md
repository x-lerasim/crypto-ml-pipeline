# crypto-ml-pipeline

End-to-end ML-пайплайн на Python + PySpark для прогнозирования цен криптовалют.

## Что делает проект

1. Загружает исторические данные по активу из CoinCap API.
2. Чистит временной ряд в Spark DataFrame.
3. Строит технические признаки (SMA, RSI, лаги, волатильность, pct_change).
4. Обучает две регрессионные модели с CrossValidator.
5. Считает RMSE, MAE, R2 и сохраняет результат в JSON.

## Стек

- Python 3.11+
- PySpark 3.5.1
- requests
- PyYAML
- pytest

## Конфигурация

Путь по умолчанию: `src/main/resources/application.yaml`.

Можно переопределить:

- через аргумент `--config`
- через переменную окружения `CRYPTOML_CONFIG_PATH`

## Локальный запуск

Требования: Python 3.11+, Java 11+, установленный Spark runtime.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python src/cryptoml_py/main.py
```

Или через spark-submit:

```bash
spark-submit --master local[*] src/cryptoml_py/main.py
```

## Docker

```bash
docker build -t crypto-ml-pipeline .
docker run --rm -v "$PWD/data:/opt/cryptoml/data" crypto-ml-pipeline
```

## Выходные артефакты

- `data/features.parquet`
- `data/metrics.json`
