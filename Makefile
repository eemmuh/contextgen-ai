.PHONY: help install dev-install env setup clean clean-models test test-unit test-integration lint format type-check security-check
.PHONY: docker-up docker-down docker-logs docker-clean
.PHONY: setup-db verify-db health-check backup-db list-backups cleanup-backups analytics migrate-faiss migrate-faiss-real
.PHONY: download-data process-data generate example
.PHONY: monitor health performance cache-info monitor-realtime export-reports optimize cache-manage
.PHONY: api-dev api-docs

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install project (editable)
	uv pip install -e .

dev-install: ## Install development dependencies
	uv pip install -e ".[dev]"

env: ## Create .env from env.example if missing
	@if [ ! -f .env ]; then cp env.example .env; echo "Created .env from env.example"; fi

setup: install env ## Install and create .env

download-data: ## Download COCO dataset subset
	python scripts/download_coco.py

process-data: ## Process dataset and create embeddings
	python -m src.main --process-dataset --dataset-type coco --max-images 1000

generate: ## Generate example images
	python -m src.main --prompt "a cat sitting on a chair" --num-images 2

example: ## Run database usage example
	python examples/database_usage.py

# Docker
docker-up: ## Start postgres + redis services (Docker)
	docker-compose -f docker/docker-compose.yml up -d postgres redis

docker-down: ## Stop all services (Docker)
	docker-compose -f docker/docker-compose.yml down

docker-logs: ## Tail Docker logs
	docker-compose -f docker/docker-compose.yml logs -f

docker-clean: ## Remove volumes and prune Docker
	docker-compose -f docker/docker-compose.yml down -v
	docker system prune -f

# Database scripts
setup-db: docker-up ## Setup PostgreSQL database and tables
	python scripts/database/setup_database.py

verify-db: ## Verify database setup
	python scripts/database/setup_database.py --verify-only

health-check: ## Run database health check
	python scripts/database/database_health_check.py

backup-db: ## Create database backup
	python scripts/database/database_backup.py backup

list-backups: ## List available backups
	python scripts/database/database_backup.py list

cleanup-backups: ## Clean up old backups (keep 30 days)
	python scripts/database/database_backup.py cleanup --keep-days 30

analytics: ## Generate database analytics report
	python scripts/database/database_analytics.py

migrate-faiss: ## Migrate from FAISS to database (dry run)
	python scripts/database/migrate_to_database.py --dry-run

migrate-faiss-real: ## Migrate from FAISS to database (actual migration)
	python scripts/database/migrate_to_database.py

# Monitoring and maintenance
monitor: ## Show system overview
	python scripts/maintenance/system_monitor.py overview

health: ## Run system health checks
	python scripts/maintenance/system_monitor.py health

performance: ## Show performance metrics
	python scripts/maintenance/system_monitor.py performance

cache-info: ## Show cache information
	python scripts/maintenance/system_monitor.py cache

monitor-realtime: ## Start real-time monitoring
	python scripts/maintenance/system_monitor.py monitor --duration 300 --interval 5

export-reports: ## Export all system reports
	python scripts/maintenance/system_monitor.py export --output-dir reports

optimize: ## Optimize system performance
	python scripts/maintenance/system_monitor.py optimize

cache-manage: ## Manage model cache
	python scripts/maintenance/cache_manager.py info

# API
api-dev: ## Run the FastAPI server in development mode
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

api-docs: ## Open API documentation in browser
	@echo "Opening API documentation at http://localhost:8000/docs"
	@open http://localhost:8000/docs || xdg-open http://localhost:8000/docs || echo "Please open http://localhost:8000/docs manually"

# Code quality
format: ## Format code with black + isort
	black src/ tests/ config/ scripts/ examples/
	isort src/ tests/ config/ scripts/ examples/

lint: ## Lint code with flake8 + mypy
	flake8 src/ tests/ config/ scripts/ examples/
	mypy src/ config/

type-check: ## Run type checking with mypy
	mypy src/ config/

security-check: ## Run security checks with bandit
	bandit -r src/ -f json -o reports/security_report.json

# Tests
test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests
	pytest tests/unit/ -v

test-integration: ## Run integration tests
	pytest tests/integration/ -v

# Cleanup
clean: ## Clean caches and build artifacts
	rm -rf logs/ reports/ .pytest_cache/ htmlcov/ dist/ build/ .coverage *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-models: ## Remove model cache directory
	rm -rf .model_cache/