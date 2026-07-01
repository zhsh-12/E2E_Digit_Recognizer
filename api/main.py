import io
import logging
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from common.preprocess import transform
from common.utils import cal_confidence, generate_csv_content
from common.database import init_db, save_single_prediction, save_batch_predictions, get_history
from api.schemas import SinglePredictionSave, BatchPredictionSave, ExportCsvRequest

base_dir = Path(__file__).resolve().parent.parent
quantized_path = base_dir/ "quantized_models"/"quantized_digit_recognizer.onnx"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

session = None

# ---------- Create lifespan manager (load model on startup, release on shutdown) ----------
@asynccontextmanager
async def lifespan(app: FastAPI):

    init_db()

    global session
    logger.info("🚀 Loading ONNX Runtime session...")

    sess_options = ort.SessionOptions() # Create session options
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # Enable all graph optimizations
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(quantized_path, sess_options, providers=providers)
   
    # Warmup: send dummy requests to avoid cold start latency
    dummy_input = np.zeros((1, 3, 28, 28), dtype=np.float32)
    for _ in range(3):
        session.run(['output'], {'input': dummy_input})
    logger.info("✅ Model loaded and warmed up successfully")
    
    yield
    # Execute on shutdown
    logger.info("👋 Shutting down...")

# ================ Create FastAPI app ================
app = FastAPI(
    title='Digit Recognizer API',
    description='Digit recognition service for universal scenario digits',
    version='1.0.0',
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server address (CORS only needed for local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================ Health check ================
@app.get('/health')
async def health():
    return {'status': 'ok', 'model_loaded': session is not None}

# ================ Single predict ================
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail='No file selected')
    
    start_time = time.perf_counter()

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess (returns numpy array of shape (1, 3, 28, 28))
        input_data = transform(image)
        
        # ONNX Runtime inference
        output = session.run(['output'], {'input': input_data})
        # output[0]: first element of model output, a numpy array of shape (1, 10)
        prediction = int(np.argmax(output[0])) # Convert numpy type to Python native type
        confidence = cal_confidence(output[0])
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"Predicted: {prediction}, confidence: {confidence:.4f}, time: {elapsed_ms:.2f}ms")
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'inference_time_ms': round(elapsed_ms, 2)
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ================ Batch predict ================
@app.post('/batch_predict')
async def batch_predict(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail='No files provided')
    
    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            input_data = transform(image)
        
            output = session.run(['output'], {'input': input_data})
            prediction = int(np.argmax(output[0]))
            confidence = cal_confidence(output[0])
            results.append({'filename': file.filename, 'prediction': prediction, 'confidence': confidence})
        
        except Exception as e:
            results.append({'filename': file.filename, 'error': str(e)})
            logger.error(f"{file.filename} failed: {str(e)}")
            
    return {'results': results}

# ================ Save results & query history ================
@app.post('/api/save_prediction')
async def api_save_prediction(data: SinglePredictionSave):
    """Save single prediction result"""
    try:
        record_id = save_single_prediction(
            predicted_label=data.predicted_label,
            confidence=data.confidence,
            filename=data.filename,
            batch_id=data.batch_id,
        )
        return {"id": record_id, "message": "Saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/save_batch_results')
async def api_save_batch_results(data: BatchPredictionSave):
    """Save batch prediction results"""
    try:
        results_dicts = [item.model_dump() for item in data.results]
        save_batch_predictions(
            batch_id=data.batch_id,
            results=results_dicts
        )
        return {"message": f"Batch {data.batch_id}: {len(data.results)} results saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/prediction_history')
async def api_prediction_history(limit: int = 50, offset: int = 0):
    """Query prediction history"""
    try:
        records = get_history(limit=limit, offset=offset)
        return {"records": records, "count": len(records)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================ CSV export API ================
@app.post('/api/export_csv')
async def api_export_csv(data: ExportCsvRequest):
    """Export prediction results as a CSV file for download"""
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

# Start backend: gunicorn api.main:app -c gunicorn.conf.py
# Start frontend: cd frontend; npm run dev