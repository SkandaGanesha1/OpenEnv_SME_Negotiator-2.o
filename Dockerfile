FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY server/requirements.txt ./server/requirements.txt
COPY README.md ./
COPY openenv.yaml ./
COPY server ./server
COPY sme_negotiator_env ./sme_negotiator_env

RUN pip install --no-cache-dir -r server/requirements.txt

ENV PORT=7860
ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
EXPOSE 7860