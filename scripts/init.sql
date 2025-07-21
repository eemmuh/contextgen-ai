-- Initialize the image_rag_db database with pgvector extension
-- This script runs automatically when the PostgreSQL container starts

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a dedicated user for the application (optional)
-- CREATE USER image_rag_user WITH PASSWORD 'your_secure_password';
-- GRANT ALL PRIVILEGES ON DATABASE image_rag_db TO image_rag_user;

-- Grant necessary permissions
GRANT ALL ON SCHEMA public TO postgres;

-- Create indexes for better performance (will be created by SQLAlchemy models)
-- These are just examples of what could be added for optimization

-- Log the initialization
DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully';
    RAISE NOTICE 'pgvector extension enabled';
    RAISE NOTICE 'Database ready for image_rag application';
END $$; 