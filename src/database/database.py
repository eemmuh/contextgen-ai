"""
Database manager for the RAG-based Image Generation System.
"""

import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .models import Base, Image, Generation, SystemMetrics
from .session import get_db_session

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database manager for handling database operations."""

    def __init__(self, database_url: str = None):
        """Initialize database manager."""
        if database_url is None:
            from config.settings import get_settings

            settings = get_settings()
            database_url = settings.database.url

        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")

    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()

    def add_image(
        self,
        image_path: str,
        description: str = None,
        tags: list = None,
        width: int = None,
        height: int = None,
        metadata: dict = None,
    ) -> int:
        """Add an image to the database."""
        with self.get_session() as session:
            image = Image(
                image_path=image_path,
                description=description,
                tags=tags or [],
                width=width,
                height=height,
                metadata=metadata or {},
            )
            session.add(image)
            session.commit()
            session.refresh(image)
            logger.info(f"Added image with ID: {image.id}")
            return image.id

    def get_image_by_id(self, image_id: int):
        """Get image by ID."""
        with self.get_session() as session:
            return session.query(Image).filter(Image.id == image_id).first()

    def get_all_images(self, limit: int = 100, offset: int = 0):
        """Get all images with pagination."""
        with self.get_session() as session:
            return session.query(Image).offset(offset).limit(limit).all()

    def delete_image(self, image_id: int) -> bool:
        """Delete an image from the database."""
        with self.get_session() as session:
            image = session.query(Image).filter(Image.id == image_id).first()
            if image:
                session.delete(image)
                session.commit()
                logger.info(f"Deleted image with ID: {image_id}")
                return True
            return False

    def add_generation(
        self,
        prompt: str,
        augmented_prompt: str = None,
        output_path: str = None,
        seed: int = None,
        generation_time_ms: int = None,
        memory_usage_mb: float = None,
        model_config: dict = None,
        retrieved_examples: list = None,
        status: str = "completed",
    ) -> int:
        """Add a generation record to the database."""
        with self.get_session() as session:
            generation = Generation(
                prompt=prompt,
                augmented_prompt=augmented_prompt,
                output_path=output_path,
                seed=seed,
                generation_time_ms=generation_time_ms,
                memory_usage_mb=memory_usage_mb,
                model_config=model_config or {},
                retrieved_examples=retrieved_examples or [],
                status=status,
            )
            session.add(generation)
            session.commit()
            session.refresh(generation)
            logger.info(f"Added generation with ID: {generation.id}")
            return generation.id

    def get_generation_by_id(self, generation_id: int):
        """Get generation by ID."""
        with self.get_session() as session:
            return session.query(Generation).filter(Generation.id == generation_id).first()

    def get_generation_history(self, limit: int = 100, status: str = None):
        """Get generation history with optional status filter."""
        with self.get_session() as session:
            query = session.query(Generation)
            if status:
                query = query.filter(Generation.status == status)
            return query.order_by(Generation.created_at.desc()).limit(limit).all()

    def add_system_metrics(
        self,
        cpu_percent: float = None,
        memory_mb: float = None,
        gpu_memory_mb: float = None,
        gpu_utilization: float = None,
        disk_usage_percent: float = None,
        cache_hit_rate: float = None,
        component: str = None,
    ) -> int:
        """Add system metrics to the database."""
        with self.get_session() as session:
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                gpu_memory_mb=gpu_memory_mb,
                gpu_utilization=gpu_utilization,
                disk_usage_percent=disk_usage_percent,
                cache_hit_rate=cache_hit_rate,
                component=component,
            )
            session.add(metrics)
            session.commit()
            session.refresh(metrics)
            return metrics.id

    def get_database_stats(self) -> dict:
        """Get database statistics."""
        with self.get_session() as session:
            # Count images
            image_count = session.query(Image).count()

            # Count generations
            generation_count = session.query(Generation).count()

            # Count system metrics
            metrics_count = session.query(SystemMetrics).count()

            # Get recent activity
            recent_generations = (
                session.query(Generation)
                .order_by(Generation.created_at.desc())
                .limit(5)
                .all()
            )

            # Get average generation time
            avg_generation_time = session.query(
                text("AVG(generation_time_ms)")
            ).scalar()

            return {
                "total_images": image_count,
                "total_generations": generation_count,
                "total_metrics": metrics_count,
                "recent_generations": len(recent_generations),
                "avg_generation_time_ms": avg_generation_time or 0,
            }

    def search_images_by_tags(self, tags: list, limit: int = 10):
        """Search images by tags."""
        with self.get_session() as session:
            # This is a simplified search - in production you might want more sophisticated matching
            query = session.query(Image)
            for tag in tags:
                query = query.filter(Image.tags.contains([tag]))
            return query.limit(limit).all()

    def get_images_by_date_range(self, start_date: datetime, end_date: datetime):
        """Get images created within a date range."""
        with self.get_session() as session:
            return (
                session.query(Image)
                .filter(Image.created_at >= start_date)
                .filter(Image.created_at <= end_date)
                .all()
            )

    def cleanup_old_metrics(self, days: int = 30):
        """Clean up old system metrics."""
        cutoff_date = datetime.now() - timedelta(days=days)
        with self.get_session() as session:
            deleted_count = (
                session.query(SystemMetrics)
                .filter(SystemMetrics.created_at < cutoff_date)
                .delete()
            )
            session.commit()
            logger.info(f"Cleaned up {deleted_count} old metrics records")
            return deleted_count
