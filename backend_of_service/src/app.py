from typing import List

from fastapi import FastAPI, HTTPException, Depends
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import PipelineModel
import uvicorn
import pandas as pd

from api_schemas import TransactionPredictionResponse, PredictResponseSchema, Transaction
from config import Config


config = Config()
app = FastAPI()
spark = SparkSession.builder \
    .appName("TransactionProcessing") \
    .master("local[*]") \
    .getOrCreate()


def load_model() -> PipelineModel:
    """
    Load the Pyspark model from the local directory.

    Returns:
        PipelineModel: model
    """
    try:
        model = PipelineModel.load(config.model_path)
        return model
    except Exception as e:
        print(f"Failed to load model from {config.model_path}: {e}")
        raise HTTPException(status_code=500, detail="Model could not be loaded")


@app.get("/api/v1/health")
async def health_check() -> dict:
    """
    Health check endpoint to verify that the application is running.
    """
    return {"status": "healthy"}


@app.get("/api/v1/ready")
async def readiness_check() -> dict:
    """
    Readiness check endpoint to ensure the application is ready to handle requests.
    """
    try:
        model = load_model()
        if model:
            return {"status": "ready"}
        else:
            return {"status": "not ready"}
    except Exception:
        return {"status": "not ready"}


@app.get("/api/v1/startup")
async def startup_check() -> dict:
    """
    Startup check endpoint to verify that the application has started successfully.
    """
    try:
        if spark:
            return {"status": "started"}
        else:
            return {"status": "not started"}
    except Exception:
        return {"status": "not started"}


@app.post("/api/v1/predict", response_model=PredictResponseSchema)
async def predict(
        transactions: List[Transaction],
        model: PipelineModel = Depends(load_model)
) -> PredictResponseSchema:
    """
    Processes the list of transactions and returns predictions.

    Args:
        transactions (List[Transaction]): List of transactions to predict.
        model (PipelineModel): The uploaded Pyspark model.

    Returns:
        PredictResponseSchema: Schema with predictions or error message.
    """
    try:
        transactions_dict = [t.dict() for t in transactions]
        transactions_df = pd.DataFrame(transactions_dict)
        transactions_df['tx_datetime'] = pd.to_datetime(transactions_df['tx_datetime'], format='%Y-%m-%dT%H:%M:%S')
        sdf: DataFrame = spark.createDataFrame(transactions_df, schema=config.schema)

        assembler = VectorAssembler(inputCols=config.vector_assembler_features,
                                    outputCol=config.vector_assembler_output_col)
        sdf = assembler.transform(sdf)
        predictions = model.transform(sdf)
        predicted_labels = [row["prediction"] for row in predictions.select("prediction").collect()]

        response = [
            TransactionPredictionResponse(transaction_id=transaction.transaction_id, prediction=pred)
            for transaction, pred in zip(transactions, predicted_labels)
        ]
        return PredictResponseSchema(data=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    config = Config()
    uvicorn.run(
        app="app:app",
        host=config.backend_host,
        port=config.backend_port,
    )