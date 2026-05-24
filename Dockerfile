# === Builder stage ===
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install normally first (to avoid PIP_TARGET issues with native libs)
RUN pip install --no-cache-dir --no-compile -r requirements.txt

# Force async-timeout — needed by redis on Python < 3.11.3 (distroless has 3.11.2)
RUN pip install --no-cache-dir --no-compile --no-deps async-timeout

# === Safe cleanup (zero risk) ===
RUN SITE=$(python -c "import site; print(site.getsitepackages()[0])") && \
    echo "Site-packages: $SITE" && \
    echo "Size before cleanup:" && du -sh $SITE && \
    # __pycache__ and bytecode
    find $SITE -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find $SITE -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true && \
    # Babel: keep only en + ru locale data (~28 MB saved)
    find $SITE/babel/locale-data -maxdepth 1 -type d ! -name "en" ! -name "ru" ! -name "locale-data" -exec rm -rf {} + 2>/dev/null || true && \
    # Remove license files
    find $SITE -type f \( -name "LICENSE*" -o -name "COPYING*" -o -name "AUTHORS*" -o -name "NOTICE*" \) -delete 2>/dev/null || true && \
    # Copy to clean target directory for distroless
    mkdir -p /dist-packages && \
    cp -r $SITE/* /dist-packages/ && \
    echo "Size after cleanup:" && du -sh /dist-packages

# === Preload all-MiniLM-L6-v2 model (force actual download) ===
RUN HOME=/tmp python -c "\
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; \
ef = ONNXMiniLM_L6_V2(); \
# Actually call to trigger model download \
result = ef(['hello world']); \
print(f'Model preloaded OK, embedding dims: {len(result[0])}')"

# === Runtime stage ===
FROM gcr.io/distroless/python3-debian12:nonroot

WORKDIR /app

# Copy cleaned site-packages to dist-packages (where distroless Python looks)
COPY --from=builder /dist-packages /usr/local/lib/python3.11/dist-packages

# Copy preloaded model cache
#COPY --from=builder --chown=65532:65532 /tmp/.cache /home/nonroot/.cache

# Copy application code
COPY --chown=65532:65532 . .

# Runtime env
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/nonroot

CMD ["main.py"]
