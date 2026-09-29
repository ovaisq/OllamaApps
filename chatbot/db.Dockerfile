FROM pgvector/pgvector:pg16
COPY pgvector.sql /docker-entrypoint-initdb.d/01-pgvector.sql
