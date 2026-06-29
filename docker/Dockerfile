# ============================================================
# Backend Dockerfile — FastAPI + PyTorch
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by Pillow, OpenCV, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone to Asia/Shanghai
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy dependency file and install Python packages
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    pillow \
    torch \
    torchvision \
    requests \
    numpy \
    pandas

# Copy application code
COPY api/ ./api/
COPY app/ ./app/
COPY common/ ./common/

# Copy trained model (read-only, bundled into image)
COPY models/ ./models/

# Expose FastAPI port
EXPOSE 8000

# Start the backend server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
