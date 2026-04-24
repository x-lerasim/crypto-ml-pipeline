# ---------- stage 1: build fat jar with sbt ----------
FROM sbtscala/scala-sbt:eclipse-temurin-11.0.22_7_1.9.9_2.12.18 AS build
WORKDIR /app

COPY project ./project
COPY build.sbt ./
# warm the cache
RUN sbt update

COPY src ./src
# use "provided" scope so spark jars are not baked in — smaller image
RUN sbt -DsparkScope=provided assembly

# ---------- stage 2: runtime on spark base ----------
FROM apache/spark:3.5.1-scala2.12-java11-ubuntu

ENV APP_HOME=/opt/cryptoml
WORKDIR $APP_HOME

COPY --from=build /app/target/scala-2.12/*assembly*.jar $APP_HOME/app.jar

USER spark
ENTRYPOINT ["/opt/spark/bin/spark-submit", \
  "--master", "local[*]", \
  "--class",  "cryptoml.Main", \
  "/opt/cryptoml/app.jar"]
