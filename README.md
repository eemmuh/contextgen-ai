# Image Model COCO Model

A RAG-based image generation system that uses COCO dataset for retrieval-augmented generation.

## Features

- **RAG-based Image Generation**: Uses COCO dataset for context-aware image generation
- **Advanced Model Caching**: Intelligent caching system with compression, validation, and monitoring
- **Multi-modal Embeddings**: CLIP and Sentence Transformers for text and image embeddings
- **FAISS Vector Search**: Fast similarity search for retrieving relevant examples
- **Stable Diffusion Integration**: High-quality image generation with context
- **Comprehensive Monitoring**: Real-time performance monitoring and health checks
- **Error Handling**: Robust error handling with retry logic and circuit breakers
- **Structured Logging**: Advanced logging with JSON output and colored console

## 🚀 Major Improvements

### 1. **Advanced Model Caching**
- **Size Limits**: Configurable memory (2GB) and disk (10GB) limits with automatic eviction
- **LRU Eviction**: Least Recently Used policy for optimal cache management
- **Compression**: Automatic compression of cached models to save disk space
- **Model Validation**: Integrity checks for cached models to ensure reliability
- **Thread Safety**: Thread-safe operations for multi-threaded applications
- **Comprehensive Monitoring**: Real-time statistics and performance metrics
- **Cache Warmup**: Preloading of frequently used models
- **Automatic Optimization**: Background cleanup of old cache entries

### 2. **Comprehensive Logging System**
- **Structured Logging**: JSON-formatted logs for easy parsing and analysis
- **Colored Console Output**: Color-coded log levels for better readability
- **Log Rotation**: Automatic log file rotation to manage disk space
- **Component-specific Loggers**: Separate loggers for different system components
- **Performance Logging**: Automatic performance metrics logging
- **Error Context**: Detailed error logging with context and stack traces

### 3. **Robust Error Handling**
- **Circuit Breaker Pattern**: Prevents cascading failures
- **Retry Logic**: Exponential backoff with configurable retry attempts
- **Error Severity Levels**: Different handling based on error severity
- **Graceful Degradation**: Fallback mechanisms for critical failures
- **Error Statistics**: Comprehensive error tracking and reporting
- **Safe Execution**: Wrapper functions for safe operation execution

### 4. **Performance Monitoring**
- **Real-time Metrics**: CPU, memory, GPU, and disk usage monitoring
- **Operation Profiling**: Detailed timing and resource usage for operations
- **Component Performance**: Per-component performance tracking
- **Memory Optimization**: Automatic memory cleanup and optimization
- **Background Monitoring**: Continuous system monitoring
- **Performance Reports**: Exportable performance metrics

### 5. **Health Check System**
- **System Health Checks**: Comprehensive system resource monitoring
- **GPU Status Monitoring**: GPU memory and utilization tracking
- **Cache Health**: Cache performance and hit rate monitoring
- **Disk Space Monitoring**: Disk usage and space availability
- **PyTorch Status**: Framework and CUDA availability checks
- **Health Reports**: Detailed health status reports

### 6. **System Management Tools**
- **Unified Monitoring Script**: Comprehensive system monitoring tool
- **Cache Management**: Advanced cache management and optimization
- **Performance Analysis**: Detailed performance analysis and reporting
- **System Optimization**: Automated system optimization
- **Report Generation**: Comprehensive system reports

## 📊 Cache Statistics

The cache provides detailed statistics including:
- Hit rate percentage
- Total requests, hits, and misses
- Memory and disk usage
- Eviction counts
- Validation failures
- Request rates

## 🛠️ System Management

### Cache Management
```bash
# View cache statistics
python scripts/cache_manager.py info

# Monitor cache performance in real-time
python scripts/cache_manager.py monitor

# Optimize cache (remove old entries)
python scripts/cache_manager.py optimize

# Warm up cache with frequently used models
python scripts/cache_manager.py warmup

# Clear specific model types
python scripts/cache_manager.py clear --model-type clip

# Clear all cache
python scripts/cache_manager.py clear

# Export cache information to JSON
python scripts/cache_manager.py export --output cache_report.json

# Reset cache statistics
python scripts/cache_manager.py reset-stats
```

### System Monitoring
```bash
# System overview
python scripts/system_monitor.py overview

# Health checks
python scripts/system_monitor.py health

# Performance metrics
python scripts/system_monitor.py performance

# Cache information
python scripts/system_monitor.py cache

# Real-time monitoring
python scripts/system_monitor.py monitor --duration 300 --interval 5

# Export all reports
python scripts/system_monitor.py export --output-dir reports

# System optimization
python scripts/system_monitor.py optimize

# All-in-one monitoring
python scripts/system_monitor.py all
```

### Makefile Commands
```bash
# System monitoring
make monitor          # System overview
make health          # Health checks
make performance     # Performance metrics
make cache-info      # Cache information
make monitor-realtime # Real-time monitoring
make export-reports  # Export all reports
make optimize        # System optimization
make cache-manage    # Cache management
```

## ⚙️ Configuration

### Cache Configuration
```python
CACHE_CONFIG = {
    "cache_dir": ".model_cache",
    "max_memory_size_mb": 2048,  # 2GB memory limit
    "max_disk_size_mb": 10240,   # 10GB disk limit
    "compression_enabled": True,
    "enable_validation": True,
    "warmup_models": [
        {
            "model_type": "sentence_transformer",
            "model_name": "all-MiniLM-L6-v2",
            "device": "cuda"
        },
        {
            "model_type": "clip",
            "model_name": "openai/clip-vit-base-patch32",
            "device": "cuda"
        }
    ],
    "optimization": {
        "auto_optimize": True,
        "optimization_interval_hours": 24,
        "max_age_days": 30
    }
}
```

### Logging Configuration
```python
# Initialize logger with custom settings
from src.utils.logger import get_logger_manager

logger_manager = get_logger_manager()
logger = logger_manager.get_logger('your_component')

# Log with context
logger.info("Operation completed", extra={
    'extra_fields': {
        'operation': 'image_generation',
        'duration_ms': 1500,
        'memory_usage_mb': 512
    }
})
```

### Error Handling
```python
from src.utils.error_handler import handle_operation, ErrorContext, ErrorSeverity

@handle_operation(ErrorContext(
    operation="model_loading",
    component="embedding",
    severity=ErrorSeverity.HIGH,
    retryable=True,
    max_retries=3
))
def load_model(model_name):
    # Your model loading code
    pass
```

### Performance Monitoring
```python
from src.utils.performance_monitor import monitor_operation

@monitor_operation("image_generation", "generation")
def generate_image(prompt):
    # Your image generation code
    pass
```

## 📈 Performance Benefits

- **Faster Startup**: Models load from cache instead of downloading
- **Reduced Bandwidth**: No repeated downloads of the same models
- **Memory Efficiency**: LRU eviction keeps most-used models in memory
- **Disk Space Optimization**: Compression and automatic cleanup
- **Reliability**: Model validation ensures cached models work correctly
- **Error Recovery**: Automatic retry and fallback mechanisms
- **Performance Insights**: Detailed performance monitoring and optimization
- **System Health**: Proactive health monitoring and alerts

## 🔧 Advanced Usage

### Cache Warmup
```python
from src.utils.model_cache import get_model_cache
from config.config import CACHE_CONFIG

cache = get_model_cache()
cache.warmup_cache(CACHE_CONFIG['warmup_models'])
```

### Custom Cache Configuration
```python
from src.utils.model_cache import ModelCache

# Create custom cache instance
custom_cache = ModelCache(
    cache_dir="custom_cache",
    max_memory_size_mb=4096,  # 4GB
    max_disk_size_mb=20480,   # 20GB
    compression_enabled=True,
    enable_validation=True
)
```

### Cache Monitoring
```python
cache = get_model_cache()
info = cache.get_cache_info()

print(f"Hit Rate: {info['hit_rate_percent']:.1f}%")
print(f"Memory Usage: {info['memory_cache_size_mb']:.1f} MB")
print(f"Disk Usage: {info['disk_cache_size_mb']:.1f} MB")
print(f"Total Requests: {info['total_requests']}")
```

### Health Monitoring
```python
from src.utils.health_check import run_health_check, get_health_summary

# Run all health checks
results = run_health_check()

# Get health summary
summary = get_health_summary()
print(f"Overall Health: {summary['overall_status']}")
```

### Performance Analysis
```python
from src.utils.performance_monitor import get_performance_summary, get_system_metrics

# Get performance summary
perf_summary = get_performance_summary()
print(f"Average Duration: {perf_summary['avg_duration']:.3f}s")

# Get system metrics
system_metrics = get_system_metrics()
print(f"Memory Usage: {system_metrics['memory']['rss_mb']:.1f} MB")
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd image-model-coco-model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the COCO dataset:
```bash
python scripts/download_coco.py
```

## Usage

### Basic Usage

```python
from src.embedding.embedding_manager import EmbeddingManager
from src.generation.image_generator import ImageGenerator

# Initialize managers (models will be cached automatically)
embedding_manager = EmbeddingManager()
image_generator = ImageGenerator()

# Generate image with context
text = "a beautiful sunset over mountains"
embedding = embedding_manager.compute_text_embedding(text)
images = image_generator.generate_image(text, num_images=1)
```

### Demo Scripts

Run the enhanced caching demo:
```bash
python examples/model_caching_demo.py
```

Run basic usage demo:
```bash
python examples/basic_usage.py
```

### System Monitoring

Check system health:
```bash
make health
```

Monitor performance:
```bash
make performance
```

Real-time monitoring:
```bash
make monitor-realtime
```

## Project Structure

```
image-model-coco-model/
├── config/                 # Configuration files
├── data/                   # Dataset storage
├── embeddings/             # Pre-computed embeddings
├── examples/               # Demo scripts
├── logs/                   # Log files (auto-generated)
├── output/                 # Generated images
├── reports/                # System reports (auto-generated)
├── scripts/                # Utility scripts
│   ├── cache_manager.py    # Cache management tool
│   ├── download_coco.py    # Dataset downloader
│   └── system_monitor.py   # System monitoring tool
├── src/                    # Source code
│   ├── data_processing/    # Dataset processing
│   ├── embedding/          # Embedding management
│   ├── generation/         # Image generation
│   ├── retrieval/          # RAG components
│   └── utils/              # Utilities
│       ├── logger.py       # Logging system
│       ├── error_handler.py # Error handling
│       ├── performance_monitor.py # Performance monitoring
│       ├── health_check.py # Health checks
│       └── model_cache.py  # Model caching
└── tests/                  # Test files
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

Run specific cache tests:
```bash
python tests/test_model_cache.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


