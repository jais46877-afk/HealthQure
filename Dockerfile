# ==============================================================================
# HealthQure Production Dockerfile (Backend Inference Service)
# Optimized for Render, Railway, Hugging Face Spaces, or GCP Cloud Run
# ==============================================================================

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install required system shared libraries for OpenCV, numpy and scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and packaging tools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies using --extra-index-url so PyPI supplies secondary dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy backend application source and model checkpoints
COPY api/ api/
COPY src/ src/
COPY checkpoints/ checkpoints/
COPY data/sample_images/ data/sample_images/
COPY data/alzheimers/ data/alzheimers/

EXPOSE 8000

# Start server dynamically binding to Render's $PORT environment variable (defaults to 8000)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
