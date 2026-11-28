# syntax=docker/dockerfile:1
# The evaluation image (F2). Build context is the REPOSITORY ROOT, because the suite reuses the
# indexer's pipeline code (extract, clean, chunk) and one pure-regex file from the API:
#     docker build -f eval/Dockerfile -t rahbar-eval:dev .
#     minikube image load rahbar-eval:dev          (make eval-image does both)
# Thresholds are NOT baked in: the Job mounts them from the ConfigMap rahbar-eval-thresholds.
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY indexer/requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install -r /tmp/requirements.txt

FROM python:3.12-slim
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp TMPDIR=/tmp
# The 2023 rules are a scan: building the offline index needs the same OCR tools as the indexer.
RUN apt-get update \
 && apt-get -y --no-install-recommends upgrade \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng poppler-utils \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin rahbar
COPY --from=build /opt/venv /opt/venv
WORKDIR /app/eval
COPY indexer/pipeline /app/indexer/pipeline
COPY api/app/retrieval/version.py /app/api/app/retrieval/version.py
COPY eval /app/eval
USER 10001
# The Job's command chooses the script (suite.py, then gate.py).
ENTRYPOINT ["python"]
CMD ["suite.py", "--help"]
