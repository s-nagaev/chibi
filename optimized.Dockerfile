# === builder ===
FROM python:3.11-alpine AS builder

# Install build dependencies
RUN apk add --no-cache build-base libffi-dev openssl-dev

WORKDIR /app
# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apk del build-base libffi-dev openssl-dev

# === runtime ===
FROM python:3.11-alpine

# Install runtime dependencies
RUN apk add --no-cache libstdc++ libffi openssl ca-certificates

WORKDIR /app
# Copy installed packages and application code
COPY --from=builder /usr/local /usr/local
COPY . .

# Create non-root user and switch
RUN addgroup -S chibi && adduser -S chibi -G chibi
USER chibi

# Launch command
CMD ["python", "main.py"]
