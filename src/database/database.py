"""
Database manager for the RAG-based Image Generation System.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from .models import Image, Embedding, Generation, ModelCache, SystemMetrics, UserSession
from .session import get_db_session_context
from src.utils.logger import get_logger

logger = get_logger("database_manager")


class DatabaseManager:
    """Main database manager for the RAG system."""
    
    def __init__(self):
        """Initialize the database manager."""
        self.logger = logger
    
    def get_db_session_context(self):
        """Get database session context."""
        from .session import get_db_session_context
        return get_db_session_context()
    
    # Image operations
    def add_image(
        self, 
        image_path: str, 
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        file_size_bytes: Optional[int] = None,
        format: Optional[str] = None,
        source_dataset: Optional[str] = None
    ) -> int:
        """Add a new image to the database."""
        with get_db_session_context() as session:
            # Check if image already exists
            existing = session.query(Image).filter(Image.image_path == image_path).first()
            if existing:
                self.logger.info(f"Image already exists: {image_path}")
                return existing.id
            
            # Create new image
            image = Image(
                image_path=image_path,
                description=description,
                tags=tags or [],
                width=width,
                height=height,
                file_size_bytes=file_size_bytes,
                format=format,
                source_dataset=source_dataset
            )
            
            session.add(image)
            session.flush()  # Get the ID
            session.refresh(image)
            
            image_id = image.id
            
            self.logger.info(f"Added image: {image_path} (ID: {image_id})")
            return image_id
    
    def get_image_by_path(self, image_path: str) -> Optional[Image]:
        """Get image by path."""
        with get_db_session_context() as session:
            return session.query(Image).filter(Image.image_path == image_path).first()
    
    def get_image_by_id(self, image_id: int) -> Optional[Image]:
        """Get image by ID."""
        with get_db_session_context() as session:
            return session.query(Image).filter(Image.id == image_id).first()
    
    def get_all_images(self, limit: Optional[int] = None) -> List[Image]:
        """Get all images with optional limit."""
        with get_db_session_context() as session:
            query = session.query(Image).order_by(desc(Image.created_at))
            if limit:
                query = query.limit(limit)
            return query.all()
    
    # Embedding operations
    def add_embedding(
        self,
        image_id: int,
        embedding: np.ndarray,
        model_type: str,
        model_name: str,
        embedding_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Embedding:
        """Add an embedding to the database."""
        with get_db_session_context() as session:
            # Check if embedding already exists for this image/model combination
            existing = session.query(Embedding).filter(
                and_(
                    Embedding.image_id == image_id,
                    Embedding.model_type == model_type,
                    Embedding.embedding_type == embedding_type
                )
            ).first()
            
            if existing:
                # Update existing embedding
                existing.embedding = embedding.tolist()
                existing.embedding_metadata = metadata
                session.flush()
                session.refresh(existing)
                self.logger.info(f"Updated embedding for image {image_id}")
                return existing
            
            # Create new embedding
            embedding_obj = Embedding.from_numpy(
                embedding=embedding,
                image_id=image_id,
                model_type=model_type,
                model_name=model_name,
                embedding_type=embedding_type,
                embedding_metadata=metadata or {}
            )
            
            session.add(embedding_obj)
            session.flush()
            session.refresh(embedding_obj)
            
            self.logger.info(f"Added embedding for image {image_id}")
            return embedding_obj
    
    def search_similar_embeddings(
        self,
        query_embedding: np.ndarray,
        model_type: str,
        embedding_type: str = "text",
        k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings using vector similarity."""
        with get_db_session_context() as session:
            # Convert query embedding to list for PostgreSQL
            query_vector = query_embedding.tolist()
            
            # Perform vector similarity search using pgvector's <-> operator
            from sqlalchemy import text
            query = text("""
                SELECT e.*, 1 - (e.embedding <=> CAST(:query_vector AS vector)) as similarity
                FROM embeddings e
                WHERE e.model_type = :model_type AND e.embedding_type = :embedding_type
                ORDER BY e.embedding <=> CAST(:query_vector AS vector)
                LIMIT :limit
            """)
            
            results = session.execute(
                query,
                {
                    "query_vector": query_vector,
                    "model_type": model_type,
                    "embedding_type": embedding_type,
                    "limit": k
                }
            ).fetchall()
            
            # Format results
            formatted_results = []
            for row in results:
                similarity = row.similarity
                if similarity >= similarity_threshold:
                    formatted_results.append({
                        'embedding_id': row.id,
                        'image_id': row.image_id,
                        'similarity_score': float(similarity),
                        'model_type': row.model_type,
                        'embedding_type': row.embedding_type,
                        'metadata': row.embedding_metadata
                    })
            
            return formatted_results
    
    def get_embeddings_for_image(self, image_id: int) -> List[Embedding]:
        """Get all embeddings for a specific image."""
        with get_db_session_context() as session:
            return session.query(Embedding).filter(Embedding.image_id == image_id).all()
    
    # Generation operations
    def add_generation(
        self,
        prompt: str,
        augmented_prompt: Optional[str] = None,
        output_path: Optional[str] = None,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        generation_time_ms: Optional[int] = None,
        memory_usage_mb: Optional[float] = None,
        model_config: Optional[Dict[str, Any]] = None,
        retrieved_examples: Optional[List[Dict[str, Any]]] = None,
        status: str = "completed",
        error_message: Optional[str] = None
    ) -> int:
        """Add a generation record to the database."""
        with get_db_session_context() as session:
            generation = Generation(
                prompt=prompt,
                augmented_prompt=augmented_prompt,
                output_path=output_path,
                seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generation_time_ms=generation_time_ms,
                memory_usage_mb=memory_usage_mb,
                model_config=model_config or {},
                retrieved_examples=retrieved_examples or [],
                status=status,
                error_message=error_message
            )
            
            session.add(generation)
            session.flush()
            session.refresh(generation)
            
            generation_id = generation.id
            self.logger.info(f"Added generation record: {generation_id}")
            return generation_id
    
    def get_generation_history(
        self, 
        limit: Optional[int] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get generation history with optional filtering."""
        with get_db_session_context() as session:
            from sqlalchemy import text
            
            query = """
                SELECT id, prompt, augmented_prompt, output_path, seed, 
                       generation_time_ms, memory_usage_mb, status, created_at
                FROM generations
                WHERE 1=1
            """
            params = {}
            
            if status:
                query += " AND status = :status"
                params['status'] = status
            
            query += " ORDER BY created_at DESC"
            
            if limit:
                query += " LIMIT :limit"
                params['limit'] = limit
            
            results = session.execute(text(query), params).fetchall()
            
            return [
                {
                    'id': row.id,
                    'prompt': row.prompt,
                    'augmented_prompt': row.augmented_prompt,
                    'output_path': row.output_path,
                    'seed': row.seed,
                    'generation_time_ms': row.generation_time_ms,
                    'memory_usage_mb': row.memory_usage_mb,
                    'status': row.status,
                    'created_at': row.created_at
                }
                for row in results
            ]
    
    # Model cache operations
    def track_model_cache(
        self,
        model_type: str,
        model_name: str,
        device: str,
        cache_key: str,
        size_bytes: Optional[int] = None,
        is_compressed: bool = False,
        validation_status: str = "valid"
    ) -> ModelCache:
        """Track model cache usage."""
        with get_db_session_context() as session:
            # Check if cache entry already exists
            existing = session.query(ModelCache).filter(
                ModelCache.cache_key == cache_key
            ).first()
            
            if existing:
                # Update existing entry
                existing.access_count += 1
                existing.last_accessed = func.now()
                existing.size_bytes = size_bytes
                existing.is_compressed = is_compressed
                existing.validation_status = validation_status
                session.flush()
                session.refresh(existing)
                return existing
            
            # Create new cache entry
            cache_entry = ModelCache(
                model_type=model_type,
                model_name=model_name,
                device=device,
                cache_key=cache_key,
                size_bytes=size_bytes,
                is_compressed=is_compressed,
                validation_status=validation_status
            )
            
            session.add(cache_entry)
            session.flush()
            session.refresh(cache_entry)
            
            return cache_entry
    
    # System metrics operations
    def add_system_metrics(
        self,
        cpu_percent: Optional[float] = None,
        memory_mb: Optional[float] = None,
        gpu_memory_mb: Optional[float] = None,
        gpu_utilization: Optional[float] = None,
        disk_usage_percent: Optional[float] = None,
        cache_hit_rate: Optional[float] = None,
        active_connections: Optional[int] = None,
        component: Optional[str] = None
    ) -> SystemMetrics:
        """Add system metrics to the database."""
        with get_db_session_context() as session:
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                gpu_memory_mb=gpu_memory_mb,
                gpu_utilization=gpu_utilization,
                disk_usage_percent=disk_usage_percent,
                cache_hit_rate=cache_hit_rate,
                active_connections=active_connections,
                component=component
            )
            
            session.add(metrics)
            session.flush()
            session.refresh(metrics)
            
            return metrics
    
    # Statistics and analytics
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with get_db_session_context() as session:
            stats = {}
            
            # Count records
            stats['total_images'] = session.query(func.count(Image.id)).scalar()
            stats['total_embeddings'] = session.query(func.count(Embedding.id)).scalar()
            stats['total_generations'] = session.query(func.count(Generation.id)).scalar()
            stats['total_cache_entries'] = session.query(func.count(ModelCache.id)).scalar()
            
            # Recent activity
            from sqlalchemy import text
            stats['recent_generations'] = session.execute(
                text("SELECT COUNT(*) FROM generations WHERE created_at >= NOW() - INTERVAL '24 hours'")
            ).scalar()
            
            # Average generation time
            avg_time = session.query(func.avg(Generation.generation_time_ms)).scalar()
            stats['avg_generation_time_ms'] = float(avg_time) if avg_time else 0
            
            return stats
    
    def cleanup_old_data(self, days: int = 30) -> Dict[str, int]:
        """Clean up old data (system metrics, old generations, etc.)."""
        with get_db_session_context() as session:
            deleted_counts = {}
            
            # Clean up old system metrics
            old_metrics = session.execute(
                text(f"DELETE FROM system_metrics WHERE timestamp < NOW() - INTERVAL '{days} days'")
            ).rowcount
            deleted_counts['system_metrics'] = old_metrics
            
            # Clean up old generations (keep successful ones longer)
            old_generations = session.execute(
                text(f"DELETE FROM generations WHERE created_at < NOW() - INTERVAL '{days} days' AND status = 'failed'")
            ).rowcount
            deleted_counts['failed_generations'] = old_generations
            
            return deleted_counts 