FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates ./templates

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
