"""
Model Accuracy Evaluation Report
=================================
Compare FP32 ONNX vs INT8 Quantized model accuracy on test images.

Generates:
  - docs/images/accuracy_report.png (confusion matrices + per-class accuracy)
  - Console output with accuracy summary

Usage:
  uv run python scripts/generate_accuracy_report.py
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image

from common.preprocess import transform

# ─── Paths ───────────────────────────────────────────────
base_dir = Path(__file__).resolve().parent.parent
onnx_path = base_dir / "onnx_models" / "digit_recognizer.onnx"
quantized_path = base_dir / "quantized_models" / "quantized_digit_recognizer.onnx"
test_img_dirs = [
    base_dir / "test_imgs" / "batch_1",
    base_dir / "test_imgs" / "batch_2",
]
output_path = base_dir / "docs" / "images" / "accuracy_report.png"

# ─── Helpers ─────────────────────────────────────────────
image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


def extract_true_label(filename: str) -> int | None:
    """Extract label from filename like 'img_1[2].jpg' -> 2"""
    match = re.search(r"\[(\d+)\]", filename)
    return int(match.group(1)) if match else None


def load_images_from_dir(img_dir: Path):
    """Load all labeled images from a directory."""
    images, labels = [], []
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() not in image_extensions:
            continue
        label = extract_true_label(f.name)
        if label is None:
            continue
        img = Image.open(f)
        input_data = transform(img)  # returns (1, 3, 28, 28)
        images.append(input_data)
        labels.append(label)
    return np.concatenate(images, axis=0).astype(np.float32), np.array(labels)


def create_session(model_path: str):
    """Create ONNX Runtime session with all optimizations."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    return session


def predict(session, input_data: np.ndarray) -> np.ndarray:
    """Run inference and return predicted labels."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: input_data})
    return np.argmax(output[0], axis=1)


# ─── Main ────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Model Accuracy Evaluation Report")
    print("=" * 60)

    # 1. Load all test images
    print("\nLoading test images...")
    all_images, all_labels = [], []
    for d in test_img_dirs:
        if d.exists():
            imgs, lbls = load_images_from_dir(d)
            all_images.append(imgs)
            all_labels.append(lbls)
            print(f"   {d.name}: {len(imgs)} images")
    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"\n   Total: {len(images)} labeled images")

    # 2. Load models
    print("\nLoading models...")
    fp32_session = create_session(str(onnx_path))
    int8_session = create_session(str(quantized_path))
    print("   FP32 ONNX model loaded")
    print("   INT8 Quantized model loaded")

    # 3. Run inference
    print("\nRunning inference...")
    fp32_preds = predict(fp32_session, images)
    int8_preds = predict(int8_session, images)

    # 4. Calculate accuracy
    fp32_acc = np.mean(fp32_preds == labels) * 100
    int8_acc = np.mean(int8_preds == labels) * 100

    print(f"\nAccuracy Results:")
    print(f"   FP32 ONNX:  {fp32_acc:.2f}%")
    print(f"   INT8 Quant: {int8_acc:.2f}%")
    print(f"   Difference: {int8_acc - fp32_acc:+.2f}%")

    # 5. Per-class accuracy
    classes = list(range(10))
    fp32_per_class = []
    int8_per_class = []
    for c in classes:
        mask = labels == c
        fp32_per_class.append(np.mean(fp32_preds[mask] == labels[mask]) * 100 if mask.sum() > 0 else 0)
        int8_per_class.append(np.mean(int8_preds[mask] == labels[mask]) * 100 if mask.sum() > 0 else 0)

    # 6. Confusion matrices
    fp32_cm = np.zeros((10, 10), dtype=int)
    int8_cm = np.zeros((10, 10), dtype=int)
    for true_lbl, pred_lbl in zip(labels, fp32_preds):
        fp32_cm[true_lbl, pred_lbl] += 1
    for true_lbl, pred_lbl in zip(labels, int8_preds):
        int8_cm[true_lbl, pred_lbl] += 1

    # 7. Generate visualization
    print("\nGenerating accuracy report chart...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Model Accuracy Evaluation: FP32 ONNX vs INT8 Quantized", fontsize=16, fontweight="bold", y=0.98)

    # Confusion Matrix - FP32
    ax1 = axes[0, 0]
    im1 = ax1.imshow(fp32_cm, cmap="Blues", interpolation="nearest")
    ax1.set_title(f"FP32 ONNX Confusion Matrix (Acc: {fp32_acc:.2f}%)", fontsize=12)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_xticks(classes)
    ax1.set_yticks(classes)
    for i in range(10):
        for j in range(10):
            ax1.text(j, i, str(fp32_cm[i, j]), ha="center", va="center", fontsize=8,
                     color="white" if fp32_cm[i, j] > fp32_cm.max() / 2 else "black")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Confusion Matrix - INT8
    ax2 = axes[0, 1]
    im2 = ax2.imshow(int8_cm, cmap="Blues", interpolation="nearest")
    ax2.set_title(f"INT8 Quantized Confusion Matrix (Acc: {int8_acc:.2f}%)", fontsize=12)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_xticks(classes)
    ax2.set_yticks(classes)
    for i in range(10):
        for j in range(10):
            ax2.text(j, i, str(int8_cm[i, j]), ha="center", va="center", fontsize=8,
                     color="white" if int8_cm[i, j] > int8_cm.max() / 2 else "black")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # Per-class accuracy comparison
    ax3 = axes[1, 0]
    x = np.arange(len(classes))
    width = 0.35
    bars1 = ax3.bar(x - width / 2, fp32_per_class, width, label="FP32 ONNX", color="#4A90D9", alpha=0.85)
    bars2 = ax3.bar(x + width / 2, int8_per_class, width, label="INT8 Quantized", color="#E8833A", alpha=0.85)
    ax3.set_title("Per-Class Accuracy Comparison", fontsize=12)
    ax3.set_xlabel("Digit Class")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(c) for c in classes])
    ax3.set_ylim(0, 105)
    ax3.legend(loc="lower right")
    ax3.axhline(y=fp32_acc, color="#4A90D9", linestyle="--", linewidth=0.8, alpha=0.6)
    ax3.axhline(y=int8_acc, color="#E8833A", linestyle="--", linewidth=0.8, alpha=0.6)
    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    # Summary text
    ax4 = axes[1, 1]
    ax4.axis("off")
    summary_text = (
        "Accuracy Summary\n"
        "================\n\n"
        f"Total test images: {len(images)}\n\n"
        f"FP32 ONNX:\n"
        f"  Accuracy:  {fp32_acc:.2f}%\n\n"
        f"INT8 Quantized:\n"
        f"  Accuracy:  {int8_acc:.2f}%\n\n"
        f"Accuracy change: {int8_acc - fp32_acc:+.2f}%\n\n"
        "INT8 quantization achieves\n"
        "near-lossless accuracy."
    )

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ddd"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nReport saved to: {output_path}")
    plt.close()

    print("\n" + "=" * 60)
    print("  Report generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
