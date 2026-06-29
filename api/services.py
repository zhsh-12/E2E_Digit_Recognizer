import torch
import io
import asyncio
import concurrent.futures
from PIL import Image
from fastapi import UploadFile
from typing import List

from app.model import get_model
from app.preprocess import set_device, transform

# Load model 
device = set_device()
model = get_model(device)

# Thread pool for parallel model inference
THREAD_POOL_WORKERS = 4
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)


# ============================================================
# tool functions
# ============================================================

async def _read_upload_file(file: UploadFile) -> dict:
    """Asynchronously read the content of a single uploaded file"""
    try:
        contents = await file.read()
        return {"filename": file.filename, "contents": contents, "error": None}
    except Exception as e:
        return {"filename": file.filename, "contents": None, "error": str(e)}

def _predict_single_image(image_bytes: bytes, filename: str) -> dict:
    """
    Run inference on a single image's byte data and return the result in dict.
    This function runs in ThreadPoolExecutor and does not block the FastAPI event loop.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
            prediction = output.argmax(dim=1).item()
        return {"filename": filename, "prediction": prediction, "error": None}
    except Exception as e:
        return {"filename": filename, "prediction": None, "error": str(e)}

async def run_inference_concurrently(files: List[UploadFile]) -> List[dict]:
    """
    Run inference on multiple images concurrently.
    First read all file contents asynchronously, then run model inference in a thread pool.
    """
    # Step 1: Asynchronously read all file contents
    read_tasks = []
    for file in files:
        read_tasks.append(_read_upload_file(file))
    file_data_list = await asyncio.gather(*read_tasks)

    # Step 2: Run inference concurrently in the thread pool
    loop = asyncio.get_event_loop()
    inference_tasks = []
    for file_data in file_data_list:
        if file_data["error"]:
            inference_tasks.append(asyncio.ensure_future(
                asyncio.sleep(0, result=file_data)
            ))
        else:
            inference_tasks.append(
                loop.run_in_executor(
                    thread_pool,
                    _predict_single_image,
                    file_data["contents"],
                    file_data["filename"],
                )
            )

    results = await asyncio.gather(*inference_tasks)
    return results
