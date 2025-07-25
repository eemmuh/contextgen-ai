"""
SQLAlchemy models for the RAG-based Image Generation System.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Image(Base):
    """Image model for storing image metadata."""

    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String(500), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    tags = Column(JSON, default=list)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Embedding(Base):
    """Embedding model for storing vector embeddings."""

    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, nullable=False, index=True)
    embedding = Column(JSON, nullable=False)  # Store as JSON array
    model_type = Column(String(100), nullable=False, index=True)
    model_name = Column(String(200), nullable=False)
    embedding_type = Column(String(50), default="text")  # text, image, etc.
    embedding_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    @classmethod
    def from_numpy(cls, embedding, image_id, model_type, model_name, embedding_type="text", embedding_metadata=None):
        """Create embedding from numpy array."""
        return cls(
            image_id=image_id,
            embedding=embedding.tolist(),
            model_type=model_type,
            model_name=model_name,
            embedding_type=embedding_type,
            embedding_metadata=embedding_metadata or {},
        )


class Generation(Base):
    """Generation model for tracking image generations."""

    __tablename__ = "generations"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    augmented_prompt = Column(Text, nullable=True)
    output_path = Column(String(500), nullable=True)
    seed = Column(Integer, nullable=True)
    generation_time_ms = Column(Integer, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    model_config = Column(JSON, default=dict)
    retrieved_examples = Column(JSON, default=list)
    status = Column(String(50), default="completed")  # completed, failed, processing
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelCache(Base):
    """Model cache tracking."""

    __tablename__ = "model_cache"

    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String(100), nullable=False, index=True)
    model_name = Column(String(200), nullable=False)
    device = Column(String(50), nullable=False)
    cache_key = Column(String(500), nullable=False, unique=True, index=True)
    size_bytes = Column(Integer, nullable=True)
    is_compressed = Column(Boolean, default=False)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    validation_status = Column(String(50), default="valid")
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemMetrics(Base):
    """System metrics tracking."""

    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)
    cpu_percent = Column(Float, nullable=True)
    memory_mb = Column(Float, nullable=True)
    gpu_memory_mb = Column(Float, nullable=True)
    gpu_utilization = Column(Float, nullable=True)
    disk_usage_percent = Column(Float, nullable=True)
    cache_hit_rate = Column(Float, nullable=True)
    component = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserSession(Base):
    """User session tracking."""

    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(200), nullable=False, unique=True, index=True)
    user_id = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    session_data = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
