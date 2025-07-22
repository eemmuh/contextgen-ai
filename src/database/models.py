"""
SQLAlchemy models for the RAG-based Image Generation System.
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, 
    ForeignKey, ARRAY, JSON, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import numpy as np
from typing import List, Dict, Any, Optional

Base = declarative_base()


class Image(Base):
    """Model for storing image metadata."""
    
    __tablename__ = "images"
    
    id = Column(Integer, primary_key=True)
    image_path = Column(String(500), nullable=False, unique=True)
    description = Column(Text)
    tags = Column(ARRAY(String))
    width = Column(Integer)
    height = Column(Integer)
    file_size_bytes = Column(Integer)
    format = Column(String(10))  # jpg, png, etc.
    source_dataset = Column(String(100))  # coco, custom, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    embeddings = relationship("Embedding", back_populates="image", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Image(id={self.id}, path='{self.image_path}', description='{self.description[:50]}...')>"


class Embedding(Base):
    """Model for storing vector embeddings with pgvector."""
    
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    embedding = Column(Vector(384), nullable=False)  # Your embedding dimension
    model_type = Column(String(50), nullable=False)  # clip, sentence_transformer, etc.
    model_name = Column(String(200), nullable=False)
    embedding_type = Column(String(20), nullable=False)  # text, image, combined
    embedding_metadata = Column(JSON)  # Additional embedding metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    image = relationship("Image", back_populates="embeddings")
    
    # Index for vector similarity search
    __table_args__ = (
        Index('idx_embedding_model_type', 'model_type'),
        Index('idx_embedding_type', 'embedding_type'),
    )
    
    def __repr__(self):
        return f"<Embedding(id={self.id}, model_type='{self.model_type}', embedding_type='{self.embedding_type}')>"
    
    @classmethod
    def from_numpy(cls, embedding: np.ndarray, **kwargs) -> 'Embedding':
        """Create embedding from numpy array."""
        return cls(embedding=embedding.tolist(), **kwargs)


class Generation(Base):
    """Model for storing image generation history."""
    
    __tablename__ = "generations"
    
    id = Column(Integer, primary_key=True)
    prompt = Column(Text, nullable=False)
    augmented_prompt = Column(Text)
    output_path = Column(String(500))
    seed = Column(Integer)
    num_inference_steps = Column(Integer)
    guidance_scale = Column(Float)
    generation_time_ms = Column(Integer)
    memory_usage_mb = Column(Float)
    model_config = Column(JSON)  # Model configuration used
    retrieved_examples = Column(JSON)  # Retrieved similar examples
    status = Column(String(20), default="completed")  # completed, failed, processing
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<Generation(id={self.id}, prompt='{self.prompt[:50]}...', status='{self.status}')>"


class ModelCache(Base):
    """Model for tracking model cache usage and statistics."""
    
    __tablename__ = "model_cache"
    
    id = Column(Integer, primary_key=True)
    model_type = Column(String(50), nullable=False)
    model_name = Column(String(200), nullable=False)
    device = Column(String(20), nullable=False)
    cache_key = Column(String(500), nullable=False, unique=True)
    size_bytes = Column(Integer)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_compressed = Column(Boolean, default=False)
    validation_status = Column(String(20), default="valid")  # valid, failed, unknown
    
    def __repr__(self):
        return f"<ModelCache(id={self.id}, model_type='{self.model_type}', model_name='{self.model_name}')>"


class SystemMetrics(Base):
    """Model for storing system performance metrics."""
    
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    cpu_percent = Column(Float)
    memory_mb = Column(Float)
    gpu_memory_mb = Column(Float)
    gpu_utilization = Column(Float)
    disk_usage_percent = Column(Float)
    cache_hit_rate = Column(Float)
    active_connections = Column(Integer)
    component = Column(String(50))  # embedding, generation, retrieval, etc.
    
    def __repr__(self):
        return f"<SystemMetrics(id={self.id}, timestamp='{self.timestamp}', component='{self.component}')>"


class UserSession(Base):
    """Model for tracking user sessions and interactions."""
    
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), nullable=False, unique=True)
    user_id = Column(String(100))  # Optional user identification
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True))
    total_generations = Column(Integer, default=0)
    total_queries = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, session_id='{self.session_id}')>" 