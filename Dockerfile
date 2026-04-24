FROM apache/spark:3.5.1-scala2.12-java11-ubuntu

ENV APP_HOME=/opt/cryptoml
WORKDIR ${APP_HOME}

COPY requirements.txt ${APP_HOME}/requirements.txt
RUN python3 -m pip install --no-cache-dir -r ${APP_HOME}/requirements.txt

COPY src ${APP_HOME}/src
COPY data ${APP_HOME}/data

ENV PYTHONPATH=${APP_HOME}/src

USER spark
ENTRYPOINT ["/opt/spark/bin/spark-submit", "--master", "local[*]", "/opt/cryptoml/src/cryptoml_py/main.py"]
