# AGENTS.md

Guidelines for AI agents working on this Whisper Live Transcription codebase.

## Project Overview

Real-time speech-to-text server with AI post-processing. Python backend serves a PWA frontend using OpenAI Whisper for transcription and Ollama for text processing (cleanup, summarization, action items).

**Architecture:**
- HTTPS (port 9090): Static file server (PWA UI)
- WSS (port 9091): Audio transcription via Whisper
- WSS (port 9092): Control channel for Ollama processing

## Build/Lint/Test Commands

**Run the server:**
```bash
python3 server.py
```

**Run with HTTP only (local dev):**
```bash
python3 server.py --http
```

**Lint with Ruff (check-only):**
```bash
ruff check --select E,W,F --ignore E501 server.py ollama_processor.py
```

**Format with Ruff (auto-fix):**
```bash
ruff format server.py ollama_processor.py
```

**Run standalone Ollama processor:**
```bash
python3 ollama_processor.py --transcript "your text here"
```

**Pre-download Whisper models:**
```bash
python3 server.py --download-models
```

**No formal test suite exists.** Test manually by:
1. Running the server: `python3 server.py --http`
2. Opening browser to `http://localhost:9090`
3. Verifying WebSocket connections

## Code Style Guidelines

### Imports

Order: stdlib → third-party → local modules:

```python
import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from whisper_live.server import TranscriptionServer
from websockets.sync.server import serve
```

- Use absolute imports
- Group imports: stdlib, third-party, local
- One import per line for clarity

### Formatting

- **Line length:** Target 88-100 characters (ruff will ignore E501)
- **Quotes:** Prefer double quotes for strings
- **Indentation:** 4 spaces (no tabs)
- **Blank lines:** 2 between top-level functions/classes, 1 within methods

### Types

Always use type hints for:
- Function parameters
- Return values (use `-> None` for None returns)
- Class attributes with dataclasses

```python
from typing import Optional, Dict, Any

def process_text(text: str, timeout: int = 300) -> Optional[Dict[str, Any]]:
    ...
```

### Naming Conventions

- **Classes:** `PascalCase` (e.g., `ControlServer`, `OllamaConfig`)
- **Functions/Methods:** `snake_case` (e.g., `process_transcript`)
- **Variables:** `snake_case` (e.g., `ssl_context`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `STATIC_DIR`, `DEFAULT_HOST`)
- **Private methods:** `_leading_underscore` (e.g., `_load_env`)

### Error Handling

- Catch specific exceptions, avoid bare `except:`
- Log errors with context before handling
- Use try/except blocks for network/external calls
- Return early on error conditions

```python
try:
    result = external_call()
except ConnectionError as e:
    logging.error(f"Failed to connect: {e}")
    return None
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    raise
```

### Documentation

- Use docstrings for classes and public methods
- Google-style format preferred
- Include Args/Returns sections for complex functions

```python
def cleanup_transcript(self, text: str) -> str:
    """Clean up and format transcribed text.
    
    Args:
        text: Raw transcription text
        
    Returns:
        Cleaned and formatted text
    """
```

### Environment & Configuration

- Both `server.py` and `ollama_processor.py` auto-load `.env` file
- Environment variables take precedence over defaults
- CLI flags override `.env` values
- Never commit `.env` files (add to `.gitignore`)

### Threading

- Use `threading.Thread` for background operations
- Use `threading.Lock()` for shared state protection
- Mark threads as `daemon=True` for automatic cleanup

### Logging

- Use the module-level logger: `logger = logging.getLogger(__name__)`
- Include context in log messages (UIDs, client info)
- Use appropriate levels: INFO for operations, DEBUG for details

### Security

- SSL/TLS is required for production (WSS/HTTPS)
- Use `html.escape()` when rendering user content
- Never log sensitive data (tokens, keys)
- Certificate files (`.pem`) should not be committed

## File Organization

```
.
├── server.py              # Main entry point (HTTP + WSS servers)
├── ollama_processor.py    # Ollama SDK wrapper (standalone)
├── cert.pem               # TLS certificate (gitignored)
├── key.pem                # TLS private key (gitignored)
├── .env                   # Environment config (gitignored)
├── env.example            # Example environment file
└── static/                # PWA frontend assets
    ├── index.html
    ├── manifest.json
    ├── sw.js
    └── icon-192.svg
```

## Dependencies

Install with pip:
```bash
pip3 install whisper-live websockets ollama faster-whisper hf-transfer
```

- `whisper-live`: WebSocket transcription server
- `websockets`: WebSocket client/server
- `ollama`: Official Ollama Python SDK
- `faster-whisper`: Optimized Whisper inference
- `hf-transfer`: (Optional) Faster HuggingFace downloads

## Quick Reference

| Task | Command |
|------|---------|
| Run server | `python3 server.py` |
| HTTP mode | `python3 server.py --http` |
| No Ollama | `python3 server.py --no-ollama` |
| Lint | `ruff check server.py ollama_processor.py` |
| Format | `ruff format server.py ollama_processor.py` |
| Download models | `python3 server.py --download-models` |

## Notes

- **No formal test suite:** Manual testing via browser required
- **No CI/CD:** Direct commits to main branch
- **Python 3.8+** required (uses dataclasses, type hints)
- **macOS/Linux:** Primary development platforms
- **iOS Support:** Requires trusted certificates via mkcert
