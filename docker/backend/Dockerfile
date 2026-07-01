# ============================================
# Backend Dockerfile - Digit Recognizer API
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set timezone to Asia/Shanghai
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies for ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file
COPY pyproject.toml .

# Install production dependencies using uv (skip dev dependencies)
# torch/torchaudio/torchvision are only needed for training, not for inference
RUN uv pip install --system --no-cache \
    "fastapi>=0.135.3" \
    "gunicorn>=26.0.0" \
    "numpy>=2.4.4" \
    "onnx>=1.21.0" \
    "onnxruntime>=1.24.4" \
    "pillow>=12.2.0" \
    "python-multipart>=0.0.32" \
    "uvicorn>=0.44.0"

# Copy application code
COPY api/ api/
COPY common/ common/

# Copy quantized model
COPY quantized_models/ quantized_models/

# Copy Gunicorn configuration
COPY gunicorn.conf.py .

# Expose port
EXPOSE 8000

# Start Gunicorn
CMD ["gunicorn", "api.main:app", "-c", "gunicorn.conf.py"]
