FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim-bookworm

LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.name="chibi"
LABEL org.label-schema.vendor="nagaev.sv@gmail.com"
LABEL org.label-schema.vcs-url="https://github.com/s-nagaev/chibi"

RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for MCP servers
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

COPY . .

# Default environment variables
ENV FILESYSTEM_ACCESS=true
ENV ENABLE_MCP_STDIO=true
ENV SKILLS_DIR=/app/skills

# Root user
ENTRYPOINT []
CMD ["python", "main.py"]
