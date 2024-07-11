from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkConf
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, TimestampType
from pyspark.sql.functions import col, hour, dayofweek, count, mean
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import mlflow
import mlflow.spark
from typing import Tuple
import numpy as np
from scipy.stats import norm


class Config:
    master = "yarn"
    app_name = "FraudDetection"
    executor_memory = "2g"
    driver_memory = "4g"
    arrow_enabled = "true"
    s3_bucket = "amamylov-mlops"
    mlflow_tracking_uri = "http://158.160.12.169:8000"
    mlflow_experiment = "fraud_detection_experiment"
    mlflow_s3_endpoint_url = "https://storage.yandexcloud.net"

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
        'tx_amount', 'tx_time_seconds', 'tx_time_days', 'tx_fraud',
        'tx_fraud_scenario'
    ]

    vector_assembler_input_old_features = [
        'customer_id', 'terminal_id', 'tx_amount', 'tx_time_seconds',
        'tx_time_days'
    ]
    vector_assembler_input_new_features = [
        'customer_id', 'terminal_id', 'tx_amount', 'tx_time_seconds',
        'tx_time_days', 'hour_of_day', 'day_of_week', 'customer_tx_count',
        'customer_avg_tx_amount', 'customer_amount_deviation',
        'terminal_tx_count', 'terminal_avg_tx_amount', 'terminal_amount_deviation'
    ]

    vector_assembler_output_col = 'features'
    label_col = 'tx_fraud'
    num_trees = 120
    max_depth = 8


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
            Tuple[SparkSession, DataFrame]: Returns Spark session and DataFrame
            with preprocessed data.
        """
        conf = (
            SparkConf().setMaster(self.config.master).setAppName(self.config.app_name)
            .set("spark.executor.memory", self.config.executor_memory)
            .set("spark.driver.memory", self.config.driver_memory)
            .set("spark.sql.execution.arrow.pyspark.enabled", self.config.arrow_enabled)
        )

        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        s3_filepath = f"s3a://{self.config.s3_bucket}/{file_name}.txt"
        sdf = spark.read.option("sep", ",").option("comment", "#").schema(
            self.config.schema).csv(s3_filepath, header=False).toDF(
            *self.config.columns)

        sdf = sdf.orderBy(col("tx_datetime"))
        sdf = sdf.na.drop(
            subset=["transaction_id", "tx_datetime", "customer_id",
                    "terminal_id", "tx_amount", "tx_time_seconds",
                    "tx_time_days", "tx_fraud"])
        sdf = sdf.dropDuplicates(['transaction_id'])
        sdf = sdf.filter(
            (col('transaction_id') >= 0) &
            (col('customer_id') >= 0) &
            (col('terminal_id') >= 0))

        output_path = f"s3a://{self.config.s3_bucket}/{file_name}.parquet"
        sdf.write.parquet(output_path, mode="overwrite")

        return spark, sdf

    def feature_engineering(self, sdf: DataFrame) -> DataFrame:
        """
        Generate new features for the dataset.

        Args:
            sdf (DataFrame): A DataFrame with preprocessed data.

        Returns:
            DataFrame: DataFrame with new features.
        """
        sdf = sdf.withColumn('day_of_week', dayofweek('tx_datetime'))
        sdf = sdf.withColumn('hour_of_day', hour('tx_datetime'))

        customer_tx_stats = sdf.groupBy('customer_id').agg(
            count('transaction_id').alias('customer_tx_count'),
            mean('tx_amount').alias('customer_avg_tx_amount')
        )
        sdf = sdf.join(customer_tx_stats, on='customer_id', how='left')

        terminal_tx_stats = sdf.groupBy('terminal_id').agg(
            count('transaction_id').alias('terminal_tx_count'),
            mean('tx_amount').alias('terminal_avg_tx_amount')
        )
        sdf = sdf.join(terminal_tx_stats, on='terminal_id', how='left')

        sdf = sdf.withColumn('customer_amount_deviation', col('tx_amount') - col('customer_avg_tx_amount'))
        sdf = sdf.withColumn('terminal_amount_deviation', col('tx_amount') - col('terminal_avg_tx_amount'))

        sdf = sdf.withColumn("customer_tx_count", sdf["customer_tx_count"].cast(IntegerType()))
        sdf = sdf.withColumn("terminal_tx_count", sdf["terminal_tx_count"].cast(IntegerType()))

        return sdf

    def evaluate_model(self, test: DataFrame, model: RandomForestClassifier) -> Tuple[float, float, float, float]:
        """
        Evaluate the model performance using precision, recall, F1-score, and AUC metrics.

        Args:
            test (DataFrame): DataFrame for predictions.
            model (RandomForestClassifier): train model for predictions.

        Returns:
            Tuple[float, float, float, float]: precision, recall, F1-score, and AUC metrics.
        """
        predictions = model.transform(test)

        precision_evaluator = MulticlassClassificationEvaluator(
            labelCol=self.config.label_col, 
            predictionCol="prediction", 
            metricName="weightedPrecision"
        )
        recall_evaluator = MulticlassClassificationEvaluator(
            labelCol=self.config.label_col, 
            predictionCol="prediction", 
            metricName="weightedRecall"
        )
        f1_evaluator = MulticlassClassificationEvaluator(
            labelCol=self.config.label_col, 
            predictionCol="prediction", 
            metricName="f1"
        )
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol=self.config.label_col, 
            rawPredictionCol="rawPrediction", 
            metricName="areaUnderROC"
        )

        precision = precision_evaluator.evaluate(predictions)
        recall = recall_evaluator.evaluate(predictions)
        f1 = f1_evaluator.evaluate(predictions)
        auc = binary_evaluator.evaluate(predictions)

        return round(precision, 5), round(recall, 5), round(f1, 5), round(auc, 5)

    def train_model(self, sdf: DataFrame, type_model: str) -> Tuple[RandomForestClassifier, DataFrame]:
        """
        Training the model using RandomForest and logging the results with MLflow.

        Args:
            sdf (DataFrame): A DataFrame with preprocessed data.
            type_model (str): Type of model to use ('new_model' or 'old_model').
            
        Returns:
            Tuple[RandomForestClassifier, DataFrame]: trained model and test sample data.
        """
        if type_model == 'new_model':
            sdf = self.feature_engineering(sdf)
            feature_names = self.config.vector_assembler_input_new_features
        elif type_model == 'old_model':
            feature_names = self.config.vector_assembler_input_old_features
        else:
            raise ValueError(f"Unsupported type_features: {type_model}")

        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment)

        with mlflow.start_run():
            assembler = VectorAssembler(inputCols=feature_names,
                                        outputCol=self.config.vector_assembler_output_col)
            sdf = assembler.transform(sdf)

            class_0 = sdf.filter(sdf.tx_fraud == 0)
            class_1 = sdf.filter(sdf.tx_fraud == 1)

            train_class_0, test_class_0 = class_0.randomSplit([0.95, 0.05], seed=42)
            train_class_1, test_class_1 = class_1.randomSplit([0.95, 0.05], seed=42)

            train = train_class_0.union(train_class_1)
            test = test_class_0.union(test_class_1)

            rf = RandomForestClassifier(labelCol=self.config.label_col,
                                        featuresCol=self.config.vector_assembler_output_col,
                                        numTrees=self.config.num_trees,
                                        maxDepth=self.config.max_depth)

            model = rf.fit(train)

            precision, recall, f1, auc = self.evaluate_model(test, model)

            mlflow.log_metrics({
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

            mlflow.log_param("num_trees", self.config.num_trees)
            mlflow.log_param("max_depth", self.config.max_depth)
            mlflow.log_param("type_model", type_model)
            mlflow.spark.log_model(model, "model")

            feature_importances = model.featureImportances.toArray()
            feature_importance_dict = {
                feature_names[i]: feature_importances[i] for i in range(len(feature_names))
            }
            sorted_feature_importances = sorted(
                feature_importance_dict.items(), key=lambda x: x[1], reverse=True
            )
            mlflow.log_text(str(sorted_feature_importances), "feature_importances.txt")

        return model, test

    def ab_test(self, test_for_old_model: DataFrame, old_model: RandomForestClassifier,
                test_for_new_model: DataFrame, new_model: RandomForestClassifier, num_bootstrap_samples: int = 100,
                alpha: float = 0.05) -> None:
        """
        A/B testing between old_model and new_model.

        Args:
            test_for_old_model (DataFrame): Test data for old model.
            old_model (RandomForestClassifier): The old version of the model.
            test_for_new_model (DataFrame): Test data for new model.
            new_model (RandomForestClassifier): The new version of the model.
            num_bootstrap_samples (int): Number of bootstrap samples to generate.
            alpha (float): Significance level for hypothesis testing.
        """
        metrics_old = {'precision': [], 'recall': [], 'f1_score': [], 'auc': []}
        metrics_new = {'precision': [], 'recall': [], 'f1_score': [], 'auc': []}

        for _ in range(num_bootstrap_samples):
            sampled_data = test_for_old_model.sample(
                withReplacement=True, fraction=1.0, seed=np.random.randint(1, 100000))

            precision_old, recall_old, f1_score_old, auc_old = self.evaluate_model(sampled_data, old_model)
            metrics_old['precision'].append(precision_old)
            metrics_old['recall'].append(recall_old)
            metrics_old['f1_score'].append(f1_score_old)
            metrics_old['auc'].append(auc_old)
            
            sampled_data = test_for_new_model.sample(
                withReplacement=True, fraction=1.0, seed=np.random.randint(1, 100000))

            precision_new, recall_new, f1_score_new, auc_new = self.evaluate_model(sampled_data, new_model)
            metrics_new['precision'].append(precision_new)
            metrics_new['recall'].append(recall_new)
            metrics_new['f1_score'].append(f1_score_new)
            metrics_new['auc'].append(auc_new)

        ci_95_metrics_old = {
            metric: (np.percentile(metrics_old[metric], 2.5), np.percentile(metrics_old[metric], 97.5))
            for metric in metrics_old
        }
        ci_95_metrics_new = {
            metric: (np.percentile(metrics_new[metric], 2.5), np.percentile(metrics_new[metric], 97.5))
            for metric in metrics_new
        }

        with mlflow.start_run():
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment)

            mlflow.log_param("num_bootstrap_samples", num_bootstrap_samples)
            mlflow.log_param("alpha", alpha)

            for metric in metrics_old:
                mlflow.log_metric(f"{metric}_old_mean", np.mean(metrics_old[metric]))
                mlflow.log_metric(f"{metric}_old_ci_2.5", ci_95_metrics_old[metric][0])
                mlflow.log_metric(f"{metric}_old_ci_97.5", ci_95_metrics_old[metric][1])

                mlflow.log_metric(f"{metric}_new_mean", np.mean(metrics_new[metric]))
                mlflow.log_metric(f"{metric}_new_ci_2.5", ci_95_metrics_new[metric][0])
                mlflow.log_metric(f"{metric}_new_ci_97.5", ci_95_metrics_new[metric][1])

            for metric in metrics_old:
                mean_old = np.mean(metrics_old[metric])
                mean_new = np.mean(metrics_new[metric])
                std_old = np.std(metrics_old[metric])
                std_new = np.std(metrics_new[metric])
                z_score = (mean_new - mean_old) / np.sqrt((std_old ** 2 + std_new ** 2) / num_bootstrap_samples)

                p_value = 2 * (1 - norm.cdf(np.abs(z_score)))

                mlflow.log_metric(f"{metric}_p_value", p_value)
                mlflow.log_metric(f"{metric}_z_score", z_score)

                if p_value < alpha:
                    mlflow.log_text(
                        f"{metric} has statistically significant difference (p_value={p_value}, alpha={alpha})",
                        f"a_b_test_results_for_{metric}.txt")
                else:
                    mlflow.log_text(
                        f"{metric} does not have statistically significant difference (p_value={p_value}, alpha={alpha})",
                        f"a_b_test_results_for_{metric}.txt")

            mlflow.end_run()


if __name__ == "__main__":
    fraud_detection = FraudDetection()
    file_name = 'small'
    spark, sdf = fraud_detection.preprocessing_data(file_name)
    old_model, test_for_old_model = fraud_detection.train_model(sdf, 'old_model')
    new_model, test_for_new_model = fraud_detection.train_model(sdf, 'new_model')
    fraud_detection.ab_test(test_for_old_model, old_model, test_for_new_model, new_model, 50, 0.05)
    spark.stop()
