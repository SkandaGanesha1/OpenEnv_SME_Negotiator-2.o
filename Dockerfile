FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-editable
COPY README.md ./
COPY openenv.yaml ./
COPY server ./server
COPY sme_negotiator_env ./sme_negotiator_env
RUN uv sync --frozen --no-editable

# Install Gradio for the interactive playground UI (not in pyproject.toml)
RUN uv pip install "gradio>=4.40.0"

COPY app.py ./

ENV PATH="/app/.venv/bin:${PATH}"

ENV PORT=7860
ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT}/ || exit 1

EXPOSE 7860

CMD ["python", "app.py"]
