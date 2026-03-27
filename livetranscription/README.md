# Whisper Live Transcription

Real-time speech-to-text with AI post-processing. Browser-based PWA using OpenAI Whisper for transcription and Ollama for summarization, cleanup, and action item extraction.

---

## Architecture

```
Browser / iOS (PWA)
    │
    ├── HTTPS  :9090 ──→ Static file server      (index.html, SW, manifest)
    ├── WSS    :9091 ──→ Whisper Live server      (audio stream → transcription)
    └── WSS    :9092 ──→ Control server           (transcript → Ollama → results)
                                                          │
                                                  Ollama :11434
                                               (http://192.168.3.16)
```

When the user stops recording, the accumulated transcript is sent to the control server on `:9092`, processed by Ollama (cleanup → summary → action items), and streamed back to the UI in real time. Multiple clients are supported simultaneously via per-client `uid` mapping.

---

## File Structure

```
.
├── server.py               # Main server (HTTP + WSS + control)
├── ollama_processor.py     # Ollama SDK wrapper (cleanup, summarize, actions)
├── cert.pem                # TLS certificate (mkcert recommended)
├── key.pem                 # TLS private key
└── static/
    ├── index.html          # PWA frontend (two-column transcription + AI panel)
    ├── manifest.json       # PWA manifest
    ├── sw.js               # Service worker (offline caching)
    └── icon-192.svg        # App icon
```

---

## Prerequisites

```bash
# Python dependencies
pip3 install whisper-live websockets ollama faster-whisper hf-transfer

# Optional: faster HuggingFace downloads
pip3 install hf-transfer
```

Ollama must be running and reachable at the configured host with at least one model pulled:

```bash
ollama pull phi4
```

---

## SSL Certificate (mkcert recommended)

Using mkcert avoids browser warnings and works on iOS without hacks. See [mkcert-setup.md](mkcert-setup.md) for full iOS trust instructions.

```bash
brew install mkcert
mkcert -install
mkcert -key-file key.pem -cert-file cert.pem 10.0.0.100 localhost 127.0.0.1
```

Place `cert.pem` and `key.pem` in the same directory as `server.py`.

Alternatively, let the server auto-generate a self-signed cert (not trusted by browsers without manual steps):

```bash
python3 server.py --generate-cert
```

---

## Configuration

All settings can be set via `.env` file, environment variables, or CLI flags. Priority order: **CLI flag > `.env` > built-in default**.

No external dependencies needed — both `server.py` and `ollama_processor.py` load `.env` natively.

```bash
cp .env.example .env
# edit .env with your values
python3 server.py
```

Add `.env` to `.gitignore`. Keep `.env.example` in version control.

### `.env` Reference

| Variable | Default | Description |
|---|---|---|
| `WHISPER_HOST` | `0.0.0.0` | Host to bind to |
| `HTTPS_PORT` | `9090` | HTTPS port for web UI |
| `WSS_PORT` | `9091` | WSS port for Whisper audio |
| `CONTROL_PORT` | `9092` | WSS port for Ollama control channel |
| `USE_HTTP` | `false` | Disable TLS (local dev only) |
| `SSL_CERT` | `cert.pem` | TLS certificate path |
| `SSL_KEY` | `key.pem` | TLS private key path |
| `WHISPER_MODEL` | `base` | Default model for pre-downloading |
| `OLLAMA_HOST` | `http://192.168.3.16` | Ollama server URL |
| `OLLAMA_MODEL` | `phi4` | Ollama model for post-processing |
| `OLLAMA_TIMEOUT` | `300` | Ollama request timeout (seconds) |
| `OLLAMA_DISABLED` | `false` | Disable Ollama control server |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Use hf-transfer for fast downloads |
| `HF_HUB_MAX_CONNECTIONS` | `16` | Parallel HuggingFace connections |

> `OLLAMA_HOST`, `OLLAMA_MODEL`, and `OLLAMA_TIMEOUT` are read by both `server.py` and `ollama_processor.py`, so running `ollama_processor.py` standalone picks up the same config.

---

### Standard (HTTPS + WSS)

```bash
python3 server.py
```

### With Ollama on a remote host

```bash
python3 server.py \
  --ollama-host http://192.168.3.16 \
  --ollama-model phi4
```

### Custom ports

```bash
python3 server.py \
  --https-port 8443 \
  --wss-port 8444 \
  --control-port 8445
```

### HTTP only (no SSL, local dev)

```bash
python3 server.py --http
```

### Without Ollama

```bash
python3 server.py --no-ollama
```

### Custom certificate paths

```bash
python3 server.py \
  --cert /path/to/cert.pem \
  --key /path/to/key.pem
```

---

## Pre-downloading Whisper Models

Models are cached in `~/.cache/huggingface/hub/` on first use. Pre-download to avoid startup delays:

```bash
# Download base, small, medium (default)
python3 server.py --download-models

# Download a specific model
python3 server.py --download-models --model large-v3
```

For faster downloads, install `hf-transfer` and set the env var:

```bash
pip3 install hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 server.py --download-models
```

Or download directly via HuggingFace for maximum speed:

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Systran/faster-whisper-large-v3', max_workers=16)
"
```

---

## CLI Reference

CLI flags override `.env` values. See Configuration section for the full env var reference.

| Argument | Env Var | Default | Description |
|---|---|---|---|
| `--host` | `WHISPER_HOST` | `0.0.0.0` | Host to bind to |
| `--https-port` | `HTTPS_PORT` | `9090` | HTTPS port for web UI |
| `--wss-port` | `WSS_PORT` | `9091` | WSS port for Whisper audio |
| `--control-port` | `CONTROL_PORT` | `9092` | WSS port for Ollama control channel |
| `--ollama-host` | `OLLAMA_HOST` | `http://192.168.3.16` | Ollama server URL |
| `--ollama-model` | `OLLAMA_MODEL` | `phi4` | Ollama model for post-processing |
| `--cert` | `SSL_CERT` | `cert.pem` | TLS certificate file |
| `--key` | `SSL_KEY` | `key.pem` | TLS private key file |
| `--http` | `USE_HTTP` | — | Disable TLS (insecure, dev only) |
| `--no-ollama` | `OLLAMA_DISABLED` | — | Disable Ollama control server |
| `--model` | `WHISPER_MODEL` | — | Model to pre-download |
| `--generate-cert` | — | — | Auto-generate self-signed cert and exit |
| `--trust-cert` | — | — | Add cert to macOS system keychain and exit |
| `--download-models` | — | — | Pre-download Whisper models and exit |

---

## Ollama Post-Processing

When recording stops, the control server runs three Ollama operations in sequence and streams each result back to the client as it completes:

| Step | Description |
|---|---|
| Cleanup | Fixes punctuation, removes filler words, formats as readable paragraphs |
| Summary | Concise paragraph summarizing key points |
| Action Items | Bulleted list of tasks, decisions, and commitments |

Results appear progressively in the right panel of the UI. Processing runs per-client in background threads — multiple simultaneous sessions are fully isolated.

To use a different model at runtime:

```bash
python3 server.py --ollama-model qwen3.5:122b
```

---

## PWA Installation

The app can be installed as a PWA on iOS and desktop:

- **iOS Safari** — tap Share → Add to Home Screen
- **Chrome/Edge** — click the install icon in the address bar or use the Install App button in the UI

The service worker caches static assets for offline access. The WebSocket connections (`:9091`, `:9092`) require network access and do not work offline.

---

## Troubleshooting

**Browser shows "Not Secure" or connection refused**
Ensure `cert.pem`/`key.pem` are present and trusted. See [mkcert-setup.md](mkcert-setup.md).

**iOS Safari cannot connect**
The mkcert root CA must be installed AND enabled under Settings → General → About → Certificate Trust Settings.

**Ollama not processing after Stop**
Check the control server is running on `:9092` and that `ollama_processor.py` is in the same directory as `server.py`. Run `python3 server.py --check-services` to verify connectivity. Ensure `--auto-enhance` is not required — post-processing triggers automatically on Stop via the control channel.

**Slow model downloads**
Install `hf-transfer` and use `snapshot_download` directly — see Pre-downloading section above.

**`static/` directory not found**
Ensure `index.html`, `manifest.json`, `sw.js`, and `icon-192.svg` are inside a `static/` folder in the same directory as `server.py`.
