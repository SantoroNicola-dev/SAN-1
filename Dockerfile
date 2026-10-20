# ==============================================================================
# SAN-1 Multimodal Content Guardrail - Production Docker Container
# ==============================================================================
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SAN1_HOST=0.0.0.0 \
    SAN1_PORT=8000 \
    SAN1_WORKERS=1 \
    PORT=8000

# Install minimal OS dependencies for ONNX Runtime and RapidOCR (OpenCV headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user
RUN useradd -m -u 1000 san1user

WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code, configuration, and quantized ONNX models
COPY config/ /app/config/
COPY models/ /app/models/
COPY s1_vision/ /app/s1_vision/
COPY README.md /app/README.md

# Install package in editable/local mode for san1 CLI command access
RUN pip install --no-cache-dir -e .

# Set permissions for non-root user
RUN chown -R san1user:san1user /app

USER san1user

# Expose standard REST API port
EXPOSE 8000

# Container Healthcheck against /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default execution: Launch production FastAPI microservice
CMD ["uvicorn", "s1_vision.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
