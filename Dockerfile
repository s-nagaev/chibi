FROM python:3.11-alpine AS builder

RUN apk update && apk upgrade && apk add --no-cache ca-certificates gcc build-base

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-alpine

LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.name="chibi"
LABEL org.label-schema.vendor="nagaev.sv@gmail.com"
LABEL org.label-schema.vcs-url="https://github.com/s-nagaev/chibi"

RUN apk update && apk upgrade --no-cache

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

COPY . .

RUN addgroup -S chibi && adduser -S chibi -G chibi
RUN chown chibi:chibi /app/data
USER chibi

ENTRYPOINT []
CMD ["python", "main.py"]
