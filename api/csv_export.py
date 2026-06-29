import csv
import io
from datetime import datetime
from pathlib import Path

from api.schemas import ExportCsvRequest


def generate_csv_content(data: ExportCsvRequest) -> tuple[str, str]:
    """
    Generate CSV content and filename.
    Returns:
        (csv_content: str, filename: str)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = io.StringIO()
    writer = csv.writer(output)

    # Determine whether it's single or batch mode
    is_batch = data.results is not None and len(data.results) > 0

    if is_batch:
        # === Batch mode ===
        has_labels = any(r.true_label is not None for r in data.results)
        batch_id = data.batch_id or "unknown"
        filename = f"batch_{batch_id}_{timestamp}.csv"

        if has_labels:
            writer.writerow(['ID', '文件名', '预测值', '真实值', '结果', '批次', '正确率', '时间'])
            for i, r in enumerate(data.results, 1):
                true_label = r.true_label if r.true_label is not None else '-'
                is_correct = '✓' if r.true_label is not None and r.predicted_label == r.true_label else '✗' if r.true_label is not None else '-'
                accuracy_str = f"{data.batch_accuracy:.1f}%" if data.batch_accuracy is not None else '-'
                writer.writerow([i, r.filename or '-', r.predicted_label, true_label, is_correct, batch_id, accuracy_str, timestamp])
        else:
            writer.writerow(['ID', '文件名', '预测值', '批次', '时间'])
            for i, r in enumerate(data.results, 1):
                writer.writerow([i, r.filename or '-', r.predicted_label, batch_id, timestamp])
    else:
        # === Single mode ===
        has_labels = data.true_label is not None
        fname = data.filename or "unknown"
        base_name = Path(fname).stem
        filename = f"predict_{base_name}_{timestamp}.csv"

        if has_labels:
            writer.writerow(['ID', '文件名', '预测值', '真实值', '结果', '批次', '正确率', '时间'])
            is_correct = '✓' if data.predicted_label == data.true_label else '✗'
            writer.writerow([1, fname, data.predicted_label, data.true_label, is_correct, '-', '-', timestamp])
        else:
            writer.writerow(['ID', '文件名', '预测值', '批次', '时间'])
            writer.writerow([1, fname, data.predicted_label, '-', timestamp])

    return output.getvalue(), filename
