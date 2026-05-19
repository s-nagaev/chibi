FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gcc build-base \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim-bookworm

LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.name="chibi"
LABEL org.label-schema.vendor="nagaev.sv@gmail.com"
LABEL org.label-schema.vcs-url="https://github.com/s-nagaev/chibi"

RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

COPY . .

# Create directories and set ownership
RUN mkdir -p /app/data /app/data/.chibi /app/home && \
    groupadd -r chibi && useradd -r -g chibi -d /app chibi && \
    chown -R chibi:chibi /app

USER chibi

ENTRYPOINT []
CMD ["python", "main.py"]
