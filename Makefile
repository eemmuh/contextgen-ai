.PHONY: help install setup clean test download-data process-data generate example monitor health optimize

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install dependencies
	uv pip install -e .

setup: install ## Set up the project (install deps + create .env)
	@if [ ! -f .env ]; then \
		echo "PEXELS_API_KEY=your_pexels_api_key_here" > .env; \
		echo "Created .env file. Please add your API keys."; \
	fi

clean: ## Clean generated files and cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	rm -rf logs/
	rm -rf reports/

test: ## Run tests
	python -m pytest tests/ -v

download-data: ## Download COCO dataset subset
	python scripts/download_coco.py

process-data: ## Process dataset and create embeddings
	python -m src.main --process-dataset --dataset-type coco --max-images 1000

generate: ## Generate example images
	python -m src.main --prompt "a cat sitting on a chair" --num-images 2

example: ## Run basic usage example
	python examples/basic_usage.py

monitor: ## Run system monitoring
	python scripts/system_monitor.py overview

health: ## Run health checks
	python scripts/system_monitor.py health

performance: ## Show performance metrics
	python scripts/system_monitor.py performance

cache-info: ## Show cache information
	python scripts/system_monitor.py cache

monitor-realtime: ## Start real-time monitoring
	python scripts/system_monitor.py monitor --duration 300 --interval 5

export-reports: ## Export all system reports
	python scripts/system_monitor.py export --output-dir reports

optimize: ## Optimize system performance
	python scripts/system_monitor.py optimize

cache-manage: ## Manage model cache
	python scripts/cache_manager.py info

format: ## Format code with black
	black src/ tests/ examples/ scripts/

lint: ## Lint code with flake8
	flake8 src/ tests/ examples/ scripts/

dev-setup: setup ## Complete development setup
	@echo "🚀 Development environment ready!"
	@echo "Next steps:"
	@echo "1. Run 'make download-data' to get COCO dataset"
	@echo "2. Run 'make process-data' to create embeddings"
	@echo "3. Run 'make example' to test the system"
	@echo "4. Run 'make monitor' to check system status" 

# Database commands
setup-db: ## Setup PostgreSQL database and tables
	python scripts/setup_database.py

verify-db: ## Verify database setup
	python scripts/setup_database.py --verify-only

health-check: ## Run database health check
	python scripts/database_health_check.py

backup-db: ## Create database backup
	python scripts/database_backup.py backup

list-backups: ## List available backups
	python scripts/database_backup.py list

cleanup-backups: ## Clean up old backups (keep 30 days)
	python scripts/database_backup.py cleanup --keep-days 30

analytics: ## Generate database analytics report
	python scripts/database_analytics.py

analytics-json: ## Generate database analytics report in JSON format
	python scripts/database_analytics.py --format json

migrate-faiss: ## Migrate from FAISS to database (dry run)
	python scripts/migrate_to_database.py --dry-run

migrate-faiss-real: ## Migrate from FAISS to database (actual migration)
	python scripts/migrate_to_database.py

list-datasets: ## List available FAISS datasets for migration 

# Docker commands
docker-up: ## Start PostgreSQL database with Docker
	docker-compose up -d postgres

docker-down: ## Stop PostgreSQL database
	docker-compose down

docker-logs: ## View PostgreSQL logs
	docker-compose logs postgres

docker-reset: ## Reset PostgreSQL database (removes all data)
	docker-compose down -v
	docker-compose up -d postgres

# Development and code quality commands
dev-install: ## Install development dependencies
	uv pip install -e ".[dev]"

type-check: ## Run type checking with mypy
	mypy src/ --ignore-missing-imports

security-check: ## Run security checks with bandit
	bandit -r src/ -f json -o reports/security-report.json

complexity-check: ## Check code complexity with radon
	radon cc src/ -a
	radon mi src/ -a

test-coverage: ## Run tests with coverage report
	pytest --cov=src --cov-report=html --cov-report=term

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	pre-commit run --all-files

clean-all: clean ## Clean everything including cache and models
	rm -rf .model_cache/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

docs-build: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs && python -m http.server 8000

# API commands
api-run: ## Run the FastAPI server
	python -m src.api.app

api-dev: ## Run the FastAPI server in development mode
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

api-docs: ## Open API documentation
	@echo "Opening API documentation at http://localhost:8000/docs"
	@open http://localhost:8000/docs || xdg-open http://localhost:8000/docs || echo "Please open http://localhost:8000/docs manually"

# Testing enhancements
test-integration-db: ## Run database integration tests
	pytest tests/test_database_integration.py -v

test-api: ## Run API tests
	pytest tests/test_api.py -v

test-all: test test-integration-db test-api ## Run all tests including integration and API tests

# Configuration management
config-validate: ## Validate configuration settings
	python -c "from config.settings import get_settings; print('Configuration is valid')"

config-show: ## Show current configuration
	python -c "from config.settings import get_settings; import json; print(json.dumps(get_settings().dict(), indent=2, default=str))"

# Complete development setup
dev-setup-complete: dev-install pre-commit-install ## Complete development environment setup
	@echo "Complete development environment ready!"
	@echo "Installed:"
	@echo "  ✅ Development dependencies"
	@echo "  ✅ Pre-commit hooks"
	@echo "  ✅ Code quality tools"
	@echo ""
	@echo "Next steps:"
	@echo "1. Run 'make docker-up' to start database"
	@echo "2. Run 'make setup-db' to setup database"
	@echo "3. Run 'make test-all' to run all tests"
	@echo "4. Run 'make lint' to check code style"
	@echo "5. Run 'make format' to format code"
	@echo "6. Run 'make api-dev' to start API server"
	@echo "7. Run 'make api-docs' to view API documentation" 



	