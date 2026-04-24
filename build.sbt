ThisBuild / scalaVersion := "2.12.18"
ThisBuild / organization := "cryptoml"
ThisBuild / version      := "0.1.0"

lazy val sparkVersion = "3.5.1"

// NOTE: for Docker / spark-submit builds switch sparkScope to "provided"
// by running: sbt -DsparkScope=provided assembly
lazy val sparkScope: String = sys.props.getOrElse("sparkScope", "compile")

lazy val root = (project in file("."))
  .settings(
    name := "crypto-ml-pipeline",
    Compile / mainClass := Some("cryptoml.Main"),
    fork := true,
    javaOptions ++= Seq(
      "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED",
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
    ),
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core"  % sparkVersion % sparkScope,
      "org.apache.spark" %% "spark-sql"   % sparkVersion % sparkScope,
      "org.apache.spark" %% "spark-mllib" % sparkVersion % sparkScope,

      "com.softwaremill.sttp.client3" %% "core"       % "3.9.7",
      "com.softwaremill.sttp.client3" %% "play-json"  % "3.9.7",
      "com.typesafe.play"             %% "play-json"  % "2.9.4",

      "com.typesafe"      % "config"       % "1.4.3",
      "ch.qos.logback"    % "logback-classic" % "1.4.14",

      "org.scalatest"    %% "scalatest"   % "3.2.18" % Test
    ),
    // Avoid assembly deduplication pain
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", xs @ _*) => MergeStrategy.discard
      case "reference.conf"              => MergeStrategy.concat
      case _                             => MergeStrategy.first
    }
  )
