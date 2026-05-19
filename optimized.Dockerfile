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

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local /usr/local
COPY . .

# Create directories first (as root)
RUN mkdir -p /app/data /app/data/.chibi /app/home

# Create user
RUN groupadd -r chibi && useradd -r -g chibi -d /app chibi

# Give ownership
RUN chown -R chibi:chibi /app

USER chibi

CMD ["python", "main.py"]