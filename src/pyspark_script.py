from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkConf
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, TimestampType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import mlflow.spark
import os
from typing import Tuple


class Config:
    master = "yarn"
    app_name = "FraudDetection"
    executor_memory = "2g"
    driver_memory = "4g"
    arrow_enabled = "true"
    s3_bucket = "amamylov-mlops"
    mlflow_tracking_uri = "http://158.160.5.43:8000"
    mlflow_experiment = "fraud_detection_experiment"
    mlflow_s3_endpoint_url = "https://storage.yandexcloud.net"
    aws_access_key_id = ""
    aws_secret_access_key = ""

    schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("tx_datetime", TimestampType(), True),
        StructField("customer_id", IntegerType(), True),
        StructField("terminal_id", IntegerType(), True),
        StructField("tx_amount", DoubleType(), True),
        StructField("tx_time_seconds", IntegerType(), True),
        StructField("tx_time_days", IntegerType(), True),
        StructField("tx_fraud", IntegerType(), True),
        StructField("tx_fraud_scenario", IntegerType(), True)
    ])

    columns = [
        'transaction_id', 'tx_datetime', 'customer_id', 'terminal_id',
        'tx_amount', 'tx_time_seconds', 'tx_time_days', 'tx_fraud', 'tx_fraud_scenario'
    ]

    vector_assembler_input_cols = [
        'customer_id', 'terminal_id', 'tx_amount', 'tx_time_seconds', 'tx_time_days'
    ]
    vector_assembler_output_col = 'features'

    label_col = 'tx_fraud'
    prediction_col = 'prediction'
    metric_name = 'f1'
    num_trees = 100


class FraudDetection:
    """
    A class for building a fraud detection model using PySpark and MlFlow.
    """

    def __init__(self) -> None:
        """
        Initialize.
        """
        self.config = Config()

    def preprocessing_data(self, file_name: str) -> Tuple[SparkSession, DataFrame]:
        """
        Data preprocessing: reading, clearing and saving in Parquet format.

        Args:
            file_name (str): The name of the file with the data to be preprocessed.

        Returns:
            Tuple[SparkSession, DataFrame]: Returns Spark session and DataFrame with preprocessed data.
        """
        conf = (
            SparkConf().setMaster(self.config.master).setAppName(self.config.app_name)
            .set("spark.executor.memory", self.config.executor_memory)
            .set("spark.driver.memory", self.config.driver_memory)
            .set("spark.sql.execution.arrow.pyspark.enabled", self.config.arrow_enabled)
        )

        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        s3_filepath = f"s3a://{self.config.s3_bucket}/{file_name}.txt"
        sdf = spark.read.option("sep", ",").option("comment", "#").schema(self.config.schema).csv(s3_filepath,
                                                                                                  header=False).toDF(
            *self.config.columns)

        sdf = sdf.orderBy(col("tx_datetime"))
        sdf = sdf.na.drop(
            subset=["transaction_id", "tx_datetime", "customer_id", "terminal_id", "tx_amount", "tx_time_seconds",
                    "tx_time_days", "tx_fraud"])
        sdf = sdf.dropDuplicates(['transaction_id'])
        sdf = sdf.filter((col('transaction_id') >= 0) & (col('customer_id') >= 0) & (col('terminal_id') >= 0))

        output_path = f"s3a://{self.config.s3_bucket}/{file_name}.parquet"
        sdf.write.parquet(output_path, mode="overwrite")

        return spark, sdf

    def train_model(self, spark: SparkSession, sdf: DataFrame) -> None:
        """
        Training the model using randomForest and logging the results with MLflow.

        Args:
            spark (SparkSession): The current Spark session.
            sdf (DataFrame): A DataFrame with preprocessed data.
        """
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.config.mlflow_s3_endpoint_url
        os.environ["AWS_ACCESS_KEY_ID"] = self.config.aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.config.aws_secret_access_key

        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment)

        with mlflow.start_run():
            assembler = VectorAssembler(inputCols=self.config.vector_assembler_input_cols,
                                        outputCol=self.config.vector_assembler_output_col)
            sdf = assembler.transform(sdf)

            train, test = sdf.randomSplit([0.8, 0.2], seed=42)

            rf = RandomForestClassifier(labelCol=self.config.label_col,
                                        featuresCol=self.config.vector_assembler_output_col,
                                        numTrees=self.config.num_trees)
            model = rf.fit(train)

            predictions = model.transform(test)

            evaluator = MulticlassClassificationEvaluator(labelCol=self.config.label_col,
                                                          predictionCol=self.config.prediction_col,
                                                          metricName=self.config.metric_name)
            f1_score = evaluator.evaluate(predictions)
            mlflow.log_metric("f1_score", f1_score)

            mlflow.spark.log_model(model, "model")

            feature_importances = model.featureImportances
            mlflow.log_text(str(feature_importances), "feature_importances.txt")


if __name__ == "__main__":
    fraud_detection = FraudDetection()
    file_name = '2019-08-22'
    spark, sdf = fraud_detection.preprocessing_data(file_name)
    fraud_detection.train_model(spark, sdf)
    spark.stop()