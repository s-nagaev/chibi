# === Builder stage ===
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir --no-compile -r requirements.txt
RUN pip install --no-cache-dir --no-compile --no-deps async-timeout

# === Safe cleanup (zero risk) ===
RUN SITE=$(python -c "import site; print(site.getsitepackages()[0])") && \
    echo "Size before cleanup:" && du -sh $SITE && \
    find $SITE -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find $SITE -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true && \
    find $SITE/babel/locale-data -maxdepth 1 -type d ! -name "en" ! -name "ru" ! -name "locale-data" -exec rm -rf {} + 2>/dev/null || true && \
    find $SITE -type f \( -name "LICENSE*" -o -name "COPYING*" -o -name "AUTHORS*" -o -name "NOTICE*" \) -delete 2>/dev/null || true && \
    echo "Size after cleanup:" && du -sh $SITE

# === Runtime stage ===
FROM python:3.11-slim-bookworm

LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.name="chibi-agent"
LABEL org.label-schema.vendor="nagaev.sv@gmail.com"
LABEL org.label-schema.vcs-url="https://github.com/s-nagaev/chibi"

RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Node.js for MCP servers
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Copy cleaned site-packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY . .

# Default environment variables
ENV FILESYSTEM_ACCESS=true
ENV ENABLE_MCP_STDIO=true
ENV SKILLS_DIR=/app/skills

CMD ["python", "main.py"]
