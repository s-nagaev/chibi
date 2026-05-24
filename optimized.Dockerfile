# === builder ===
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# === runtime ===
FROM python:3.11-slim-bookworm
