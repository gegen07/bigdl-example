ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.10"
val SparkVersion = "3.1.3"

lazy val root = (project in file("."))
  .settings(
    name := "dl-spark",
    libraryDependencies ++= Seq(
      "com.intel.analytics.bigdl" % "bigdl-dllib-spark_3.1.3" % "2.2.0" % Provided,
      "org.apache.spark" %% "spark-core" % SparkVersion % Provided,
      "org.apache.spark" %% "spark-sql" % SparkVersion % Provided,
      "org.apache.spark" %% "spark-mllib" % "2.4.0" % Compile,
    ),
  )
