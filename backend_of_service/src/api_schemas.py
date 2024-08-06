from typing import Optional, List
from pydantic import BaseModel


class Transaction(BaseModel):
    """Schema for representing a single transaction in the input data."""
    transaction_id: int
    tx_datetime: str
    customer_id: int
    terminal_id: int
    tx_amount: float
    tx_time_seconds: int
    tx_time_days: int


class TransactionPredictionResponse(BaseModel):
    """Schema for representing the prediction result for a transaction."""
    transaction_id: int
    prediction: float
    verdict: str


class PredictResponseSchema(BaseModel):
    """Schema for the response from the prediction endpoint."""
    data: Optional[List[TransactionPredictionResponse]] = None
    error: Optional[str] = None
