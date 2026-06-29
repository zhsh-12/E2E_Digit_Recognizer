import requests
from common.config import predict_url, batch_predict_url, request_timeout


def get_prediction(image_path: str) -> int:
    """Send a single image to predict service and return the prediction"""
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(predict_url, files=files, timeout=request_timeout)
        response.raise_for_status()
        result = response.json()
        return result['prediction']


def get_batch_predictions(image_paths: list) -> list:
    """
    Send multiple images to batch_predict service and return the predictions.

    Args:
        image_paths: List of image file paths

    Returns:
        list[dict]: Each element is {"filename": str, "prediction": int|null, "error": str|null}
    """
    file_objs = []
    try:
        for path in image_paths:
            file_objs.append(('files', open(path, 'rb')))

        response = requests.post(batch_predict_url, files=file_objs, timeout=request_timeout)
        response.raise_for_status()
        return response.json()['results']
    finally:
        for _, f in file_objs:
            f.close()
