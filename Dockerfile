FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# One `uv sync` after all Hatch package sources exist on disk (sme_negotiator_env, server, rl).
# Explicit altair/pandas install is a safety net if the Space builds from a stale `uv.lock`.
COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY openenv.yaml ./
COPY server ./server
COPY sme_negotiator_env ./sme_negotiator_env
COPY rl ./rl
COPY outputs/judge_ui ./outputs/judge_ui
COPY app.py config.py action_handler.py reward_engine.py session_store.py step_logger.py ./

RUN uv sync --frozen --no-editable \
    && uv pip install "altair>=5.0.0" "pandas>=2.0.0" \
    && .venv/bin/python -c "import altair, pandas, gradio; print('Gradio UI deps OK')"

ENV PATH="/app/.venv/bin:${PATH}"

ENV PORT=7860
ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT}/ || exit 1

EXPOSE 7860

CMD ["python", "app.py"]
