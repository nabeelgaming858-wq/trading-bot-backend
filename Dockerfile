# Use Python 3.11 slim — stable, compatible with gunicorn + Cloud Run
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY static/ ./static/

# Cloud Run injects PORT env var automatically (default 8080)
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Single worker, sync type — most stable for Cloud Run free tier
# Using 1 worker avoids memory issues on small instances
CMD exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --worker-class sync \
    --timeout 120 \
    --keep-alive 5 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    app:app
