from typing import List, Optional
from pydantic import BaseModel

class SinglePredictionSave(BaseModel):
    predicted_label: int
    confidence: float
    filename: Optional[str] = None
    batch_id: Optional[str] = None


class BatchResultItem(BaseModel):
    filename: Optional[str] = None
    predicted_label: int
    confidence: float

class BatchPredictionSave(BaseModel):
    batch_id: str
    results: List[BatchResultItem]


class ExportCsvRequest(BaseModel):
    # Single prediction mode
    filename: Optional[str] = None
    predicted_label: Optional[int] = None
    confidence: Optional[float] = None
    # Batch prediction mode
    batch_id: Optional[str] = None
    results: Optional[List[BatchResultItem]] = None
