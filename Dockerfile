# Multi-stage Dockerfile for Disney RAG System

FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY config/ config/
COPY eval/ eval/
COPY Makefile .

# Copy data and indices (in production, these would be in volumes)
COPY data/indices/ data/indices/
COPY data/processed/ data/processed/
COPY data/lookup/ data/lookup/

# Create logs directory
RUN mkdir -p logs

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

