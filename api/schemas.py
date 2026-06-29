from typing import List, Optional
from pydantic import BaseModel

class SinglePredictionSave(BaseModel):
    predicted_label: int
    true_label: Optional[int] = None
    filename: Optional[str] = None
    session_id: Optional[str] = None


class BatchResultItem(BaseModel):
    filename: Optional[str] = None
    predicted_label: int
    true_label: Optional[int] = None


class BatchPredictionSave(BaseModel):
    batch_id: str
    results: List[BatchResultItem]
    batch_accuracy: Optional[float] = None
    session_id: Optional[str] = None


class ExportCsvRequest(BaseModel):
    # Single prediction scenario
    filename: Optional[str] = None
    predicted_label: Optional[int] = None
    true_label: Optional[int] = None
    # Batch prediction scenario
    batch_id: Optional[str] = None
    results: Optional[List[BatchResultItem]] = None
    batch_accuracy: Optional[float] = None
