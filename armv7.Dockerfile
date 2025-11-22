FROM python:3.11-slim AS builder

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
       build-essential gcc libffi-dev libxml2-dev libxslt-dev cargo rustc zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

RUN echo "deb http://deb.debian.org/debian bookworm main" > /etc/apt/sources.list.d/bookworm.list \
    && echo "deb http://security.debian.org/debian-security bookworm-security main" > /etc/apt/sources.list.d/security.list \
    && echo "deb http://deb.debian.org/debian trixie main" > /etc/apt/sources.list.d/unstable.list \
    && printf "Package: *\nPin: release a=trixie\nPin-Priority: 90\n" > /etc/apt/preferences.d/unstable.pref \
    && apt-get update \
    && apt-get dist-upgrade -y \
    && apt-get install -y --no-install-recommends -t trixie libxml2 zlib1g \
    && apt-get install -y --no-install-recommends libffi8 libxslt1.1 \
    && rm -rf /etc/apt/sources.list.d/unstable.list /etc/apt/preferences.d/unstable.pref /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

COPY . .

RUN addgroup --system chibi \
    && adduser --system --ingroup chibi chibi \
    && mkdir -p /app/data \
    && chown -R chibi:chibi /app/data

USER chibi

ENTRYPOINT []
CMD ["python", "main.py"]