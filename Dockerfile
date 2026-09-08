FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md batcher.py ./
COPY inference ./inference
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/inference/.local/bin:$PATH

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin inference
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

USER inference
WORKDIR /home/inference
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "inference.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
