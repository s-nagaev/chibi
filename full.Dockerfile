# Unified Chibi image: chibi backend (agent mode) + chibi-tui terminal client.
# The TUI spawns `chibi stdio --tui`, so the backend console script is installed
# in the same image; skills are served from /app/skills.
#
# The TUI source is supplied through a named build context (`tui-src`), declared
# below as an empty scratch stage and overridden at build time.
#
# Local checkout (the TUI repository is not yet published):
#   docker buildx build \
#     --build-context tui-src=/Users/sergio/Develop/personal/chibi-tui \
#     -f full.Dockerfile -t pysergio/chibi:full .
#
# Remote git context (once the TUI repository is pushed):
#   docker buildx build \
#     --build-context tui-src=https://github.com/s-nagaev/chibi-tui \
#     -f full.Dockerfile -t pysergio/chibi:full .

# === TUI source (empty placeholder, replaced by --build-context tui-src=...) ===
FROM scratch AS tui-src

# === TUI builder ===
FROM rust:slim-bookworm AS tui-builder

WORKDIR /tui
COPY --from=tui-src Cargo.toml Cargo.lock ./
COPY --from=tui-src src ./src

RUN cargo build --release

# === Backend builder ===
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY chibi ./chibi
COPY skills ./skills
# data/.keep is excluded from the build context by .dockerignore; recreate it
# so the poetry includes resolve during wheel build.
RUN mkdir -p data && touch data/.keep

RUN pip install --no-cache-dir --no-compile -r requirements.txt
RUN pip install --no-cache-dir --no-compile --no-deps async-timeout
RUN pip install --no-cache-dir --no-compile --no-deps .

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
LABEL org.label-schema.name="chibi-full"
LABEL org.label-schema.vendor="nagaev.sv@gmail.com"
LABEL org.label-schema.vcs-url="https://github.com/s-nagaev/chibi"

RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Node.js for MCP servers
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Copy cleaned site-packages and console scripts (chibi)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the TUI binary (glibc-compatible: rust:slim-bookworm -> python:slim-bookworm)
COPY --from=tui-builder /tui/target/release/chibi-tui /usr/local/bin/chibi-tui

WORKDIR /app
COPY skills ./skills
RUN mkdir -p /app/data

# Default environment variables (agent mode)
ENV FILESYSTEM_ACCESS=true
ENV ENABLE_MCP_STDIO=true
ENV SKILLS_DIR=/app/skills

CMD ["chibi-tui"]
