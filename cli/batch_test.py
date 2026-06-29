import time
from pathlib import Path
from common.config import batch_size
from common.database import init_cli_db, save_batch_results
from cli.utils import extract_true_label, detect_label_mode, get_image_files
from cli.client import get_batch_predictions

base_dir = Path(__file__).resolve().parent.parent
img_path = base_dir / "input_imgs" 
img_path.mkdir(parents=True, exist_ok=True)


def run_batch(batch_id: str, img_path: Path):
    """
    Run a batch test.

    Auto-detection mode:
    - If filenames contain [number] formatted true labels → labeled mode (calculates accuracy)
    - If filenames do not contain [number] formatted true labels → unlabeled mode (outputs predictions only)

    Notes:
    - Max batch_size per request
    - Leverages backend /batch_predict concurrent inference for faster processing
    - Removed time.sleep intervals
    """
    # Ensure database table exists
    init_cli_db()

    image_files = get_image_files(img_path=img_path)
    if not image_files:
        print("No image files found. Please check the input_imgs folder")
        return

    total_count = len(image_files)
    print(f"Batch {batch_id} started, {total_count} images in total (batch size: {batch_size})")

    # Auto-detect whether true labels exist
    has_labels = detect_label_mode(image_files)
    if has_labels:
        print("Detected true labels in filenames → Labeled mode (accuracy will be calculated)")
    else:
        print("No true labels detected → Unlabeled mode (predictions only)")

    all_results = []
    correct = 0
    total = 0

    start = time.time()

    # Process in  batch_size
    for i in range(0, total_count, batch_size):
        batch = image_files[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_count + batch_size - 1) // batch_size

        print(f"\n--- Batch {batch_num}/{total_batches} ({len(batch)} images) ---")

        try:
            predictions = get_batch_predictions(batch)
        except Exception as e:
            print(f"Batch prediction request failed: {e}")
            continue

        for img_file, pred_result in zip(batch, predictions):
            total += 1
            
            filename = img_file.name
            pred_label = pred_result.get("prediction")
            error = pred_result.get("error")

            if error:
                print(f"{filename}: Prediction failed - {error}")
                continue

            if pred_label is None:
                print(f"{filename}: Empty prediction result")
                continue

            if has_labels:
                try:
                    true_label = extract_true_label(filename)
                except ValueError as e:
                    print(f"{filename}: {e}")
                    continue
            else:
                true_label = None

            if has_labels:
                is_correct = (pred_label == true_label)
                if is_correct:
                    correct += 1
                status = "✓" if is_correct else "✗"
                print(f"{filename}: pred={pred_label}, true={true_label} {status}")
            else:
                print(f"{filename}: pred={pred_label}")

            all_results.append((filename, pred_label, true_label))

    elapsed = time.time() - start

    if has_labels:
        batch_accuracy = (correct / total * 100) if total > 0 else 0.0
        print(f"\nBatch {batch_id} completed: {total} images, {correct} correct, accuracy: {batch_accuracy:.2f}%, elapsed: {elapsed:.4f}s")
    else:
        batch_accuracy = None
        print(f"\nBatch {batch_id} completed: {total} images (unlabeled mode, predictions only), elapsed: {elapsed:.4f}s")

    if all_results:
        save_batch_results(batch_id, all_results, batch_accuracy)
    else:
        print("No valid results to save")

if __name__ == "__main__":
    batch_id = "batch_2"
    img_path = img_path / batch_id
    run_batch(batch_id, img_path)

# Run: cd cli; uv run batch_test.py [set batch_id first, update batch_id and img_path accordingly]
# Note: Make sure the backend is running first: uvicorn api.main:app --reload --port 8000
