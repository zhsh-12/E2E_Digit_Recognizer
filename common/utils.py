import csv
import io
import numpy as np
from datetime import datetime
from pathlib import Path

from api.schemas import ExportCsvRequest

def cal_confidence(ndarray):
    """Calculate confidence from model output (shape: (1, 10))"""
    logits = ndarray  
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    confidence = float(np.max(probs[0]))
    return confidence

def generate_csv_content(data: ExportCsvRequest) -> tuple[str, str]:
    """Generate CSV content and filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = io.StringIO()
    writer = csv.writer(output)

    is_batch = data.results is not None and len(data.results) > 0

    if is_batch:
        # === Batch mode ===
        batch_id = data.batch_id or "unknown"
        filename = f"batch_predict_{batch_id}_{timestamp}.csv"

        writer.writerow(['ID', '批次', '文件名', '预测值', '置信度', '时间'])
        for i, r in enumerate(data.results, 1): # Start enumeration from 1 (default is 0)
            writer.writerow([i, batch_id, r.filename or '-', r.predicted_label, r.confidence, timestamp])
    else:
        # === Single mode ===
        fname = data.filename or "unknown"
        base_name = Path(fname).stem
        filename = f"predict_{base_name}_{timestamp}.csv"

        writer.writerow(['ID', '批次', '文件名', '预测值', '置信度', '时间'])
        writer.writerow([1, '-', fname, data.predicted_label, data.confidence, timestamp])

    return output.getvalue(), filename