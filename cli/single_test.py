import time
from pathlib import Path
from cli.client import get_prediction

base_dir = Path(__file__).resolve().parent.parent
img_path = base_dir / "input_imgs" 
img_path.mkdir(parents=True, exist_ok=True)

def run_predict(img_path):
    start = time.time()
    prediction = get_prediction(img_path)
    elapsed = time.time() - start
    print(f"prediction: {prediction}")  # e.g. prediction: 2
    print(f"{elapsed:.4f}s") #e.g. elapsed: 0.0144s

if __name__ == "__main__":
    batch_id = "batch_1"
    img_file = "img_1[2].jpg"
    img_path = img_path / batch_id / img_file
    run_predict(img_path)

# Note: Make sure the backend is running first: uvicorn api.main:app --reload --port 8000
