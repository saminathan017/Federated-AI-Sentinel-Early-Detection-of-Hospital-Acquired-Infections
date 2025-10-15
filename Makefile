# Federated AI Sentinel Makefile
# Run 'make help' to see all available commands

.PHONY: help setup lint test train_baseline train_temporal eval explain serve_api serve_ui run_kafka run_prom_grafana federated_sim autopush clean

# Default target shows help
help:
	@echo "Federated AI Sentinel - Available Commands"
	@echo ""
	@echo "Setup and Development:"
	@echo "  make setup              Install all dependencies and initialize services"
	@echo "  make lint               Run code formatters and linters (Ruff, Black, Mypy)"
	@echo "  make test               Run full test suite with coverage"
	@echo "  make clean              Remove temporary files and caches"
	@echo ""
	@echo "Data and Training:"
	@echo "  make generate_data      Generate synthetic hospital datasets"
	@echo "  make train_baseline     Train XGBoost baseline model"
	@echo "  make train_temporal     Train temporal transformer model"
	@echo "  make eval               Evaluate trained models"
	@echo "  make explain            Generate SHAP explanations"
	@echo ""
	@echo "Federated Learning:"
	@echo "  make federated_sim      Simulate three-site federated training"
	@echo ""
	@echo "Serving:"
	@echo "  make serve_api          Start FastAPI inference server"
	@echo "  make serve_ui           Start React development server"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make run_kafka          Start Kafka/Redpanda for event streaming"
	@echo "  make run_prom_grafana   Start Prometheus and Grafana monitoring"
	@echo "  make run_mlflow         Start MLflow tracking server"
	@echo "  make run_db             Start PostgreSQL database"
	@echo ""
	@echo "Automation:"
	@echo "  make autopush           Watch for changes and auto-commit to GitHub"
	@echo ""

# Setup: Install dependencies and initialize
setup:
	@echo "Installing Python dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Installing Node dependencies for UI..."
	cd src/ui/clinician_app && npm install
	@echo "Creating necessary directories..."
	mkdir -p data models logs mlruns prometheus_data grafana_data kafka_data
	@echo "Initializing Feast feature store..."
	cd src/features/feast_repo && feast apply
	@echo "Setup complete. Run 'make help' to see available commands."

# Code quality
lint:
	@echo "Running Ruff linter..."
	ruff check src/ tests/ tools/
	@echo "Running Black formatter..."
	black src/ tests/ tools/
	@echo "Running Mypy type checker..."
	mypy src/ tests/ tools/

# Testing
test:
	@echo "Running unit tests..."
	pytest tests/unit/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "Running end-to-end tests..."
	pytest tests/e2e/ -v
	@echo "Coverage report saved to htmlcov/index.html"

# Data generation
generate_data:
	@echo "Generating synthetic hospital datasets..."
	python -m src.data.simulators.generate_synthetic_hospital_data

# Training
train_baseline:
	@echo "Training XGBoost baseline model..."
	python -m src.modeling.baselines.xgb_baseline

train_temporal:
	@echo "Training temporal transformer model..."
	python -m src.modeling.temporal.train

# Evaluation
eval:
	@echo "Evaluating trained models..."
	python -m src.modeling.temporal.evaluate

# Explainability
explain:
	@echo "Generating SHAP explanations..."
	python -m src.explainability.shap_explainer

# Federated learning
federated_sim:
	@echo "Starting federated learning simulation with three hospital sites..."
	python -m src.federated.simulate_three_sites

# API serving
serve_api:
	@echo "Starting FastAPI inference server..."
	uvicorn src.serving.api.main:app --host 0.0.0.0 --port 8000 --reload

# UI serving
serve_ui:
	@echo "Starting React development server..."
	cd src/ui/clinician_app && npm run dev

# Infrastructure services
run_kafka:
	@echo "Starting Redpanda (Kafka-compatible) in Docker..."
	docker run -d --name redpanda \
		-p 9092:9092 \
		-p 9644:9644 \
		docker.redpanda.com/redpandadata/redpanda:latest \
		redpanda start --smp 1 --memory 1G --reserve-memory 0M \
		--overprovisioned --node-id 0 --check=false \
		--kafka-addr internal://0.0.0.0:9092,external://0.0.0.0:19092 \
		--advertise-kafka-addr internal://redpanda:9092,external://localhost:19092
	@echo "Redpanda running on localhost:9092"

run_prom_grafana:
	@echo "Starting Prometheus..."
	docker run -d --name prometheus \
		-p 9090:9090 \
		-v $(PWD)/src/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
		-v $(PWD)/prometheus_data:/prometheus \
		prom/prometheus:latest
	@echo "Starting Grafana..."
	docker run -d --name grafana \
		-p 3000:3000 \
		-v $(PWD)/grafana_data:/var/lib/grafana \
		-e "GF_SECURITY_ADMIN_USER=admin" \
		-e "GF_SECURITY_ADMIN_PASSWORD=admin" \
		grafana/grafana:latest
	@echo "Prometheus running on localhost:9090"
	@echo "Grafana running on localhost:3000 (admin/admin)"

run_mlflow:
	@echo "Starting MLflow tracking server..."
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

run_db:
	@echo "Starting PostgreSQL database..."
	docker run -d --name postgres \
		-p 5432:5432 \
		-e POSTGRES_USER=sentinel \
		-e POSTGRES_PASSWORD=sentinel \
		-e POSTGRES_DB=sentinel_db \
		-v $(PWD)/data/postgres:/var/lib/postgresql/data \
		postgres:15-alpine
	@echo "PostgreSQL running on localhost:5432"

# Autopush tool
autopush:
	@echo "Starting autopush watcher..."
	python tools/autopush.py

# Cleanup
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf htmlcov/ .coverage
	@echo "Stopping Docker containers..."
	docker stop redpanda prometheus grafana postgres 2>/dev/null || true
	docker rm redpanda prometheus grafana postgres 2>/dev/null || true
	@echo "Cleanup complete."

