FROM python:3.11-slim

WORKDIR /app

# Install gcc for any compiled packages
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY static/ ./static/

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

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
