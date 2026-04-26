.PHONY: help install install-dev test test-fast lint format clean server docker-build docker-run baseline diagnostic examples judge-pack

help:
	@echo "OpenEnv SME Negotiator - Development Commands"
	@echo ""
	@echo "  make install       pip install -e ."
	@echo "  make test          pytest tests/"
	@echo "  make server        uvicorn on 0.0.0.0:7860"
	@echo "  make baseline      python inference.py"
	@echo "  make judge-pack    build judge-facing artifacts from inference_results.json"
	@echo "  make diagnostic    import check + pytest"
	@echo "  make docker-build / docker-run"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -q

test-cov:
	pytest tests/ --cov=sme_negotiator_env --cov=server --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

lint:
	python -m compileall -q server sme_negotiator_env tests

format:
	@echo "Optional: pip install ruff && ruff format server sme_negotiator_env tests"

typecheck:
	@echo "Optional: pip install mypy && mypy server sme_negotiator_env"

clean:
	python -c "import pathlib, shutil; \
[shutil.rmtree(p, True) for p in pathlib.Path('.').rglob('__pycache__')]; \
[shutil.rmtree(p, True) for p in ['build','dist','htmlcov','.pytest_cache','.mypy_cache','.ruff_cache'] if pathlib.Path(p).exists()]; \
[shutil.rmtree(p, True) for p in pathlib.Path('.').glob('*.egg-info')]"

server:
	python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

examples:
	@echo "No examples/ folder yet; see README Quick Start and inference.py"

baseline:
	python inference.py

judge-pack:
	python -m rl.judge_pack --results-file inference_results.json --output-dir outputs/judge_pack

diagnostic:
	python -c "from server.app import app; print('app:', app.title)"
	pytest tests/ -q --tb=line

docker-build:
	docker build -f docker/Dockerfile -t openenv-sme-negotiator:latest .

docker-run:
	docker run -p 7860:7860 openenv-sme-negotiator:latest

docker-push:
	@echo "docker tag && docker push (set your registry)"

docs:
	@echo "README.md, SETUP.md, EVALUATION.md, TROUBLESHOOTING.md"

.PHONY: all
all: install-dev test
