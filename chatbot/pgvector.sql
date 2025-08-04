CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE markdown_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk TEXT NOT NULL,
    embedding vector(1024) NOT NULL,
    metadata JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX markdown_chunks_embedding_hnsw_index 
ON markdown_chunks USING hnsw (embedding vector_l2_ops);
