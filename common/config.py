from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
db_path = base_dir / "digit_recognizer.db"
predict_url = 'http://localhost:8000/predict'
batch_predict_url = 'http://localhost:8000/batch_predict'
image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
request_timeout = 30 # usually in second
batch_size = 50
