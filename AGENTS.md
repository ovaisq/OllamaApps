# AGENTS.md

Guidelines for AI agents working on this OllamaApps codebase.

## Project Overview

This is a monorepo of locally-hosted AI applications using Ollama for LLM inference.

**Projects:**
- `chatbot/` - Markdown chatbot (ChromaDB + pgvector variants)
- `CodeReviewAssistant/` - Gradio-based code review UI
- `codey/` - Command-line code review tool
- `embedding/` - Web document Q&A with PostgreSQL/pgvector
- `livetranscription/` - Real-time Whisper transcription with Ollama post-processing

## Build/Lint/Test Commands

**Run individual projects:**
```bash
# Live Transcription Server
cd livetranscription && python3 server.py --http

# Embedding Web Assistant
cd embedding && python3 ui.py

# Code Review UI
cd CodeReviewAssistant && python3 ui.py

# Chatbot
cd chatbot && python3 chromadb_chatty.py
```

**Lint with Ruff:**
```bash
# Check all Python files
ruff check . --select E,W,F --ignore E501

# Check specific files
ruff check server.py ollama_processor.py

# Auto-fix issues
ruff format .
ruff check --fix .
```

**Docker builds:**
```bash
cd <project> && ./build_n_deploy.sh
```

**No formal test suite exists.** Test manually by running the application.

## Code Style Guidelines

### Imports

Order: stdlib → third-party → local modules. Group imports with blank lines:

```python
import argparse
import logging
import os
from typing import Optional, List, Dict, Any

import gradio as gr
import ollama
from langchain_core.documents import Document

from config import get_config
from websearch import create_dict_list_from_text
```

- Use absolute imports
- One import per line for clarity
- Place imports that may fail (optional deps) in try/except blocks

### Formatting

- **Line length:** Target 88-100 characters (ignore E501 for longer lines)
- **Quotes:** Prefer double quotes for strings
- **Indentation:** 4 spaces (no tabs)
- **Blank lines:** 2 between top-level functions/classes, 1 within methods

### Types

Use type hints for:
- Function parameters
- Return values (`-> None` for None returns)
- Class attributes

```python
from typing import Optional, Dict, Any

def process_text(text: str, timeout: int = 300) -> Optional[Dict[str, Any]]:
    ...
```

### Naming Conventions

- **Classes:** `PascalCase` (e.g., `OllamaService`, `FileProcessor`)
- **Functions/Methods:** `snake_case` (e.g., `process_transcript`)
- **Variables:** `snake_case` (e.g., `ssl_context`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `DEFAULT_HOST`, `CHAT_MODEL`)
- **Private methods:** `_leading_underscore` (e.g., `_load_env`)

### Error Handling

- Catch specific exceptions, avoid bare `except:`
- Log errors with context before handling
- Use try/except for network/external calls

```python
try:
    result = external_call()
except ConnectionError as e:
    logger.error(f"Failed to connect: {e}")
    return None
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

### Documentation

- Use docstrings for classes and public methods
- Google-style format preferred
- Include Args/Returns for complex functions

```python
def retrieve_context(query: str, k: int = 3) -> List[str]:
    """Retrieve relevant documents from ChromaDB.
    
    Args:
        query: User's query text
        k: Number of top results to retrieve
        
    Returns:
        List of document texts retrieved as context
    """
```

### Configuration

- Use `.env` files for environment variables (auto-load via `_load_env()`)
- Never commit `.env` files
- Provide `env.example` or `config.py.template`
- Environment variables take precedence over defaults

### Logging

- Use module-level logger: `logger = logging.getLogger(__name__)`
- Include context in log messages (UIDs, client info)
- Use appropriate levels: INFO for operations, DEBUG for details

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
```

### Security

- Never log sensitive data (tokens, keys)
- Use `html.escape()` when rendering user content
- Certificate files (`.pem`) should not be committed

## Dependencies

Common dependencies across projects:
- `ollama` - Official Ollama Python SDK
- `gradio` - Web UI framework
- `chromadb` - Vector database
- `psycopg2-binary` - PostgreSQL adapter
- `langchain*` - LLM framework components

Install per-project requirements:
```bash
pip3 install -r requirements.txt
```

## File Organization

```
.
├── chatbot/              # Chatbot implementations
├── CodeReviewAssistant/  # Code review UI
├── codey/                # CLI code review
├── embedding/            # Web document Q&A
├── livetranscription/    # Whisper + Ollama
├── docs/                 # Documentation
└── AGENTS.md            # This file
```

## Quick Reference

| Task | Command |
|------|---------|
| Lint | `ruff check . --select E,W,F` |
| Format | `ruff format .` |
| Run project | `cd <dir> && python3 <main>.py` |
| Docker build | `./build_n_deploy.sh` |

## Notes

- **Python 3.8+** required
- **No formal tests:** Manual testing required
- **No CI/CD:** Direct commits to main
- Each subproject has independent requirements.txt
