import io
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List

from app.model import get_model
from app.preprocess import set_device, transform
from common.database import init_frontend_db, save_single_prediction, save_batch_frontend, get_frontend_history
from api.schemas import SinglePredictionSave, BatchPredictionSave, ExportCsvRequest
from api.services import run_inference_concurrently
from api.csv_export import generate_csv_content

MAX_BATCH_SIZE = 50  # Maximum images per batch prediction
MAX_REQUEST_SIZE_MB = 100  # Maximum request body size (MB)

#Step 1: Load model 
device = set_device()
model = get_model(device)

#Step 2: create backend service
app = FastAPI(
    title="Digit Recognition API",
    description="Digit recognition service for universal scenario digits",
    max_request_size=MAX_REQUEST_SIZE_MB * 1024 * 1024,
)

#Step 3: initialize database for logging
init_frontend_db()

#Step 4: add CORS middleware (allow Vue frontend cross-origin access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Health check
# ============================================================
@app.get('/health')
def health():
    return {'status': 'ok'}


# ============================================================
# Single prediction
# ============================================================
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """
    Single prediction: accepts 1 image, returns the prediction
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)) 
        image_tensor = transform(image).unsqueeze(0).to(device) 

        with torch.no_grad():
            output = model(image_tensor)
            prediction = output.argmax(dim=1).item()
        return {'prediction': prediction}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Batch prediction
# ============================================================
@app.post('/batch_predict')
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Batch prediction: accepts multiple images, returns predictions for each file

    Features:
    - Max MAX_BATCH_SIZE images per request (default 50)
    - Uses thread pool for concurrent inference, significantly improving speed
    - Asynchronously reads files without blocking the event loop
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files selected")

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch prediction supports at most {MAX_BATCH_SIZE} images per request, "
                    f"but {len(files)} were uploaded. "
                    f"Please upload in batches of no more than {MAX_BATCH_SIZE}."
        )

    try:
        results = await run_inference_concurrently(files)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# ============================================================
# Save results & query history
# ============================================================

@app.post('/api/save_prediction')
async def api_save_prediction(data: SinglePredictionSave):
    """Save single prediction result"""
    try:
        record_id = save_single_prediction(
            predicted_label=data.predicted_label,
            true_label=data.true_label,
            filename=data.filename,
            session_id=data.session_id,
        )
        return {"id": record_id, "message": "Saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/save_batch_results')
async def api_save_batch_results(data: BatchPredictionSave):
    """Save batch prediction results"""
    try:
        results_dicts = [item.model_dump() for item in data.results]
        save_batch_frontend(
            batch_id=data.batch_id,
            results=results_dicts,
            batch_accuracy=data.batch_accuracy,
            session_id=data.session_id,
        )
        return {"message": f"Batch {data.batch_id}: {len(data.results)} results saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/prediction_history')
async def api_prediction_history(limit: int = 50, offset: int = 0):
    """Query frontend prediction history"""
    try:
        records = get_frontend_history(limit=limit, offset=offset)
        return {"records": records, "count": len(records)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# CSV export API
# ============================================================

@app.post('/api/export_csv')
async def api_export_csv(data: ExportCsvRequest):
    """
    Export prediction results as a CSV file download.
    Auto-detects mode:
    - Having true labels: CSV includes [ID, Filename, Prediction, True Label, Result, Batch, Accuracy, Time]
    - No true labels: CSV includes [ID, Filename, Prediction, Batch, Time]
    """
    try:
        csv_content, filename = generate_csv_content(data)
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "text/csv; charset=utf-8-sig",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")


# Start backend: uvicorn api.main:app --reload --port 8000
# Start frontend: cd frontend; npm run dev
