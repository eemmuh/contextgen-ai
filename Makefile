.PHONY: help install setup clean test download-data process-data generate example monitor health optimize

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

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