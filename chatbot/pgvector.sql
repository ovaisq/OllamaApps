-- 1024 = the output dimension of the default EMBEDDING_MODEL
-- (qwen3-embedding:0.6b). Must stay in sync with EMBEDDING_DIM in
-- pgv_config.py.template.
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

-- Both tables are also created/migrated idempotently at app startup
-- (pgv_schema.py / pgv_chatty.py), so this file only matters for a
-- from-scratch install; an already-running deployment self-migrates
-- without needing this file re-run.
CREATE TABLE IF NOT EXISTS chat_history (
    id BIGSERIAL PRIMARY KEY,
    user_email TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS chat_history_user_email_idx
ON chat_history (user_email, created_at);

-- Like/Dislike feedback on assistant answers; the Admin tab's
-- "Teach a correction" panel turns disliked Q&As into fixed knowledge.
CREATE TABLE IF NOT EXISTS feedback (
    id BIGSERIAL PRIMARY KEY,
    user_email TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
