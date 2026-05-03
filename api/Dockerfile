FROM python:3.11-slim

# System deps for psycopg2 and LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY api/ api/
COPY database/ database/
COPY frontend/ frontend/

# Model storage directory
RUN mkdir -p /app/models

ENV MODEL_DIR=/app/models \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

EXPOSE ${PORT}

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
