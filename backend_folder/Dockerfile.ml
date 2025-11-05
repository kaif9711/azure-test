FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ML
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libopencv-dev \
    python3-opencv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Additional ML dependencies
RUN pip install --no-cache-dir \
    scikit-learn==1.3.0 \
    opencv-python==4.8.0.74 \
    pillow==10.0.0 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    tensorflow==2.13.0

# Copy ML models and related code
COPY models/ ./models/
COPY utils/ ./utils/
COPY config.py .

# Create necessary directories
RUN mkdir -p logs uploads

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models

# Create non-root user
RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
USER mluser

# Expose port for ML service
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Create a simple ML service runner
RUN echo '#!/usr/bin/env python3\n\
from flask import Flask, jsonify\n\
import sys\n\
import os\n\
sys.path.append("/app")\n\
\n\
app = Flask(__name__)\n\
\n\
@app.route("/health")\n\
def health():\n\
    return jsonify({"status": "healthy", "service": "ml-service"})\n\
\n\
@app.route("/predict", methods=["POST"])\n\
def predict():\n\
    # ML prediction logic would go here\n\
    return jsonify({"prediction": "fraud", "confidence": 0.85})\n\
\n\
if __name__ == "__main__":\n\
    app.run(host="0.0.0.0", port=8001, debug=False)\n\
' > /app/ml_service.py && chmod +x /app/ml_service.py

# Run the ML service
CMD ["python", "ml_service.py"]