-- Create extension for vector support
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table to store markdown chunks and embeddings
CREATE TABLE markdown_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk TEXT NOT NULL,
    embedding vector(1024) NOT NULL,  -- Assuming 1024-dimensional embeddings from the model
    metadata JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index on embedding column for similarity search using l2 operator class
CREATE INDEX markdown_chunks_embedding_hnsw_index 
ON markdown_chunks USING hnsw (embedding vector_l2_ops);
