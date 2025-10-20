.PHONY: help install ingest build-index serve serve-ui eval test test-unit test-integration lint clean docker-build docker-up docker-down docker-logs

help:
	@echo "Disney RAG System - Available Commands:"
	@echo "  make install          - Install all dependencies"
	@echo "  make ingest           - Run data ingestion and feature extraction"
	@echo "  make build-index      - Build FAISS and BM25 indices"
	@echo "  make serve            - Start FastAPI service"
	@echo "  make serve-ui         - Start Streamlit UI"
	@echo "  make eval             - Run evaluation on gold dataset"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make lint             - Run code linters (black, flake8, mypy)"
	@echo "  make clean            - Clean generated files and caches"
	@echo "  make docker-build     - Build Docker images"
	@echo "  make docker-up        - Start all services with Docker Compose"
	@echo "  make docker-down      - Stop all Docker services"
	@echo "  make docker-logs      - View Docker logs"

install:
	pip3 install --upgrade pip
	pip3 install -r requirements.txt
	@echo "✅ Dependencies installed successfully"

ingest:
	python3 -m src.ingestion.pipeline
	@echo "✅ Data ingestion completed"

build-index:
	python3 -m src.retrieval.build_indices
	@echo "✅ Indices built successfully"

serve:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

serve-ui:
	python3 -m streamlit run src/ui/app.py --server.port 8501

eval:
	python3 -m eval.evaluator
	@echo "✅ Evaluation completed"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-unit:
	pytest tests/unit/ -v --cov=src --cov-report=term-missing

test-integration:
	pytest tests/integration/ -v --cov=src --cov-report=term-missing

lint:
	black --check src/ tests/ eval/
	flake8 src/ tests/ eval/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ eval/

clean:
	rm -rf data/processed/* data/indices/* data/cache/* logs/* eval/results/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "✅ Cleaned generated files and caches"

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "✅ All services started"
	@echo "API: http://localhost:8000"
	@echo "UI: http://localhost:8501"
	@echo "Prometheus: http://localhost:9090"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

