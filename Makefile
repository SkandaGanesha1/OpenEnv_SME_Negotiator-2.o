.PHONY: help install test lint format clean server docker docs

help:
	@echo "OpenEnv SME Negotiator - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies"
	@echo "  make install-dev      Install with dev tools"
	@echo "  make diagnostic       Verify installation and connectivity"
	@echo ""
	@echo "Development:"
	@echo "  make test             Run tests"
	@echo "  make test-fast        Run tests with minimal output"
	@echo "  make lint             Run code quality checks"
	@echo "  make format           Auto-format code"
	@echo "  make typecheck        Run type checking"
	@echo ""
	@echo "Running:"
	@echo "  make server           Start FastAPI server"
	@echo "  make baseline         Run baseline inference with OpenAI (requires OPENAI_API_KEY)"
	@echo "  make examples         Run example scripts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run Docker container"
	@echo "  make docker-push      Push image to registry"
	@echo ""
	@echo "Utility:"
	@echo "  make clean            Remove build artifacts"
	@echo "  make docs             Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -q

test-cov:
	pytest tests/ --cov=src --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

lint:
	@echo "Checking code style..."
	black --check src/ tests/ examples/
	ruff check src/ tests/
	mypy src/ 2>/dev/null || true

format:
	@echo "Formatting code..."
	black src/ tests/ examples/
	ruff check src/ tests/ --fix

typecheck:
	mypy src/ --ignore-missing-imports

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf htmlcov/ .coverage .pytest_cache/
	rm -rf .mypy_cache/ .ruff_cache/

server:
	python -m uvicorn src.server:app --reload --port 8000

examples:
	python examples/01_basic_usage.py

baseline:
	python inference.py

diagnostic:
	python run_diagnostics.py

docker-build:
	docker build -f docker/Dockerfile -t openenv-sme-negotiator:latest .

docker-run:
	docker run -p 8000:8000 openenv-sme-negotiator:latest

docker-push:
	@read -p "Enter registry (e.g., docker.io/username): " REGISTRY; \
	docker tag openenv-sme-negotiator:latest $$REGISTRY/openenv-sme-negotiator:latest; \
	docker push $$REGISTRY/openenv-sme-negotiator:latest

docs:
	@echo "Documentation:"
	@echo "  - README.md                  Main documentation"
	@echo "  - DEVELOPMENT.md             Developer guide"
	@echo "  - HF_SPACES_DEPLOYMENT.md   HF Spaces deployment"
	@echo "  - NEMOTRON_INTEGRATION.md   Baseline agent integration"

.PHONY: all
all: install-dev lint test
