import json
from datetime import datetime
import threading

from kafka import KafkaConsumer
from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, TimestampType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel
import mlflow.spark
import boto3


class Config:
    master: str = "yarn"
    app_name: str = "InferenceModels"
    executor_memory: str = "2g"
    driver_memory: str = "4g"
    arrow_enabled: str = "true"
    s3_bucket: str = "amamylov-mlops"
    mlflow_tracking_uri: str = "http://158.160.8.194:8000"
    mlflow_experiment: str = "fraud_detection_experiment"
    mlflow_s3_endpoint_url: str = "https://storage.yandexcloud.net"
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

    vector_assembler_features: list = [
        'customer_id', 'terminal_id', 'tx_amount', 'tx_time_seconds',
        'tx_time_days'
    ]
    vector_assembler_output_col: str = 'features'

    group_id: str = "test"
    bootstrap_server: str = "rc1b-1l3e8h0kmhs5hdun.mdb.yandexcloud.net:9091"
    user: str = "amamylov"
    password: str = ""
    topic: str = "test_topic"
    run_id: str = "7b118bd42d084d02b9028e9df6174a7b"
    timer: int = 600


class InferenceModels:
    def __init__(self) -> None:
        self.config = Config()
        self.spark = self.create_spark_session()
        self.model = self.load_model()
        self.output_file = self.generate_output_filename()
        self._stop_event = threading.Event()
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.config.aws_access_key_id,
            aws_secret_access_key=self.config.aws_secret_access_key,
            endpoint_url=self.config.mlflow_s3_endpoint_url
        )
        self.consumer = KafkaConsumer(
            bootstrap_servers=self.config.bootstrap_server,
            security_protocol="SASL_SSL",
            sasl_mechanism="SCRAM-SHA-512",
            sasl_plain_username=self.config.user,
            sasl_plain_password=self.config.password,
            ssl_cafile="YandexCA.crt",
            group_id=self.config.group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        )

    @staticmethod
    def generate_output_filename() -> str:
        """
        Generate an output filename based on the current timestamp.

        Returns:
            str: Generated filename.
        """
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"inference_{current_time}.json"

    def upload_to_s3(self) -> None:
        """
        Upload the output file to the configured S3 bucket.
        """
        try:
            self.s3_client.upload_file(self.output_file, self.config.s3_bucket, self.output_file)
            print(f"File {self.output_file} uploaded to S3 bucket {self.config.s3_bucket}")
        except Exception as e:
            print(f"Failed to upload file {self.output_file}")

    def create_spark_session(self) -> SparkSession:
        """
        Create a Spark session with specified configurations.

        Returns:
            SparkSession: Initialized Spark session.
        """
        conf = (
            SparkConf().setMaster(self.config.master).setAppName(self.config.app_name)
            .set("spark.executor.memory", self.config.executor_memory)
            .set("spark.driver.memory", self.config.driver_memory)
            .set("spark.sql.execution.arrow.pyspark.enabled", self.config.arrow_enabled)
        )

        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        return spark

    def load_model(self) -> RandomForestClassificationModel:
        """
        Load the RandomForestClassificationModel from MLflow.

        Returns:
            RandomForestClassificationModel: Loaded model.
        """
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment)
        model_uri = f"s3://{self.config.s3_bucket}/artifacts/1/{self.config.run_id}/artifacts/model"
        return mlflow.spark.load_model(model_uri)

    def predict(self, message: dict) -> float:
        """
        Perform prediction using the loaded model on a single message.

        Args:
            message (dict): Input message containing transaction details.

        Returns:
            float: Predicted fraud probability.
        """
        message['tx_datetime'] = datetime.strptime(message['tx_datetime'], '%Y-%m-%dT%H:%M:%S')
        sdf: DataFrame = self.spark.createDataFrame([message], schema=self.config.schema)
        assembler = VectorAssembler(inputCols=self.config.vector_assembler_features,
                                    outputCol=self.config.vector_assembler_output_col)
        sdf = assembler.transform(sdf)
        predictions = self.model.transform(sdf)
        prediction = predictions.select("prediction").collect()[0]["prediction"]
        return prediction

    def thread_stop_consumer(self) -> None:
        """
        Stop the Kafka consumer and upload the output file to S3 after the timer expires.
        """
        print(f"Stopping consumer after {self.config.timer} seconds")
        self._stop_event.set()
        self.upload_to_s3()
        if self.consumer:
            self.consumer.close()
        self.spark.stop()

    def upload_to_s3(self) -> None:
        try:
            self.s3_client.upload_file(self.output_file, self.config.s3_bucket, self.output_file)
            print(f"File {self.output_file} uploaded to S3 bucket {self.config.s3_bucket}")
        except Exception as e:
            print(f"Failed to upload file to S3: {e}")

    def consume_messages(self) -> None:
        """
        Consume messages from Kafka topic, perform prediction, and print results.
        """
        self.consumer.subscribe([self.config.topic])

        timer = threading.Timer(self.config.timer, self.thread_stop_consumer)
        timer.start()

        print("Waiting for new messages. Press Ctrl+C to stop")
        count = 0
        try:
            with open(self.output_file, 'a') as file:
                for msg in self.consumer:
                    if self._stop_event.is_set():
                        break
                    transaction = msg.value
                    prediction = self.predict(transaction)
                    transaction['prediction'] = prediction
                    transaction['tx_datetime'] = transaction['tx_datetime'].strftime('%Y-%m-%dT%H:%M:%S')
                    file.write(json.dumps(transaction) + '\n')
                    print(f"Transaction ID: {transaction['transaction_id']}, Prediction: {prediction}")
                    count += 1
        except KeyboardInterrupt:
            print(f"Total {count} messages received")
        finally:
            if not self._stop_event.is_set():
                self.upload_to_s3()
            self.consumer.close()
            self.spark.stop()


if __name__ == "__main__":
    predictor = InferenceModels()
    predictor.consume_messages()