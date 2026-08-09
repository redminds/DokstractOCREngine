FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r dokstract && useradd -r -g dokstract -d /app -s /sbin/nologin dokstract \
    && mkdir -p /app/.data /app/logs /app/.data/ocr-cache \
    && chown -R dokstract:dokstract /app

COPY requirements.lock.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.lock.txt

COPY app ./app

# Source fingerprint for deterministic freshness proof
# Computed at build time from app/*.py files (excluding pycache, tests)
RUN python -c "import hashlib; from pathlib import Path; \
files=sorted(p for p in Path('app').rglob('*.py') if '__pycache__' not in str(p) and '/tests/' not in str(p)); \
content=''.join(open(str(f)).read() for f in files); \
Path('/app/.source_fingerprint').write_text(hashlib.sha256(content.encode()).hexdigest())" \
    && chown dokstract:dokstract /app/.source_fingerprint

LABEL com.dokstract.ocr-engine.source-fingerprint-file=/app/.source_fingerprint

EXPOSE 8010

USER dokstract

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
