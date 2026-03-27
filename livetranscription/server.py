#!/usr/bin/env python3
"""Whisper Live transcription server with HTTPS/WSS support and built-in web UI."""

import argparse
import html
import logging
import os
import re
import ssl
import socket
import threading
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from whisper_live.server import TranscriptionServer
from websockets.sync.server import serve


# Load .env file if present (must happen before reading env vars)
def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(
                        key.strip(), val.strip().strip('"').strip("'")
                    )


_load_env()

# Enable hf-transfer for faster model downloads (if installed)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_MAX_CONNECTIONS", "16")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Path to static files directory
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class MarkdownConverter:
    """Convert markdown text to HTML for display in the UI."""

    def __init__(self):
        self.rules = [
            (
                r"\*\*\*([^*]+)\*\*\*",
                r"<strong><em>\1</em></strong>",
            ),  # ***bold italic***
            (r"\*\*([^*]+)\*\*", r"<strong>\1</strong>"),  # **bold**
            (r"\*([^*]+)\*", r"<em>\1</em>"),  # *italic*
            (r"`([^`]+)`", r"<code>\1</code>"),  # `code`
            (r"~~([^~]+)~~", r"<del>\1</del>"),  # ~~strikethrough~~
            (r"^#{6}\s+(.+)$", r"<h6>\1</h6>"),  # ###### h6
            (r"^#{5}\s+(.+)$", r"<h5>\1</h5>"),  # ##### h5
            (r"^#{4}\s+(.+)$", r"<h4>\1</h4>"),  # #### h4
            (r"^#{3}\s+(.+)$", r"<h3>\1</h3>"),  # ### h3
            (r"^#{2}\s+(.+)$", r"<h2>\1</h2>"),  # ## h2
            (r"^#\s+(.+)$", r"<h1>\1</h1>"),  # # h1
        ]

    def convert(self, text: str) -> str:
        """Convert markdown text to HTML."""
        if not text or not text.strip():
            return ""

        # Escape HTML first
        html_text = html.escape(text)

        # Apply markdown rules
        for pattern, replacement in self.rules:
            html_text = re.sub(pattern, replacement, html_text, flags=re.MULTILINE)

        # Handle links [text](url)
        html_text = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            r'<a href="\2" target="_blank">\1</a>',
            html_text,
        )

        # Handle unordered lists
        html_text = self._convert_lists(html_text)

        # Convert newlines to <br> or paragraphs
        html_text = self._convert_paragraphs(html_text)

        return html_text

    def _convert_lists(self, text: str) -> str:
        """Convert markdown lists to HTML."""
        lines = text.split("\n")
        result = []
        in_list = False
        list_type = None  # 'ul' or 'ol'

        for line in lines:
            # Unordered list
            ul_match = re.match(r"^(\s*)[-*+]\s+(.+)$", line)
            # Ordered list
            ol_match = re.match(r"^(\s*)\d+\.\s+(.+)$", line)

            if ul_match:
                if not in_list:
                    result.append("<ul>")
                    in_list = True
                    list_type = "ul"
                elif list_type != "ul":
                    result.append(f"</{list_type}><ul>")
                    list_type = "ul"
                result.append(f"<li>{ul_match.group(2)}</li>")
            elif ol_match:
                if not in_list:
                    result.append("<ol>")
                    in_list = True
                    list_type = "ol"
                elif list_type != "ol":
                    result.append(f"</{list_type}><ol>")
                    list_type = "ol"
                result.append(f"<li>{ol_match.group(2)}</li>")
            else:
                if in_list:
                    result.append(f"</{list_type}>")
                    in_list = False
                    list_type = None
                result.append(line)

        if in_list:
            result.append(f"</{list_type}>")

        return "\n".join(result)

    def _convert_paragraphs(self, text: str) -> str:
        """Convert text to paragraphs, preserving block elements."""
        lines = text.split("\n")
        result = []
        current_para = []

        block_elements = ("<h1", "<h2", "<h3", "<h4", "<h5", "<h6", "<ul", "<ol", "<li")

        for line in lines:
            stripped = line.strip()

            # If line starts with a block element, flush current paragraph first
            if stripped.startswith(block_elements):
                if current_para:
                    result.append("<p>" + " ".join(current_para) + "</p>")
                    current_para = []
                result.append(line)
            elif not stripped:
                # Empty line ends paragraph
                if current_para:
                    result.append("<p>" + " ".join(current_para) + "</p>")
                    current_para = []
            else:
                current_para.append(line)

        # Flush remaining paragraph
        if current_para:
            result.append("<p>" + " ".join(current_para) + "</p>")

        return "\n".join(result)


@dataclass
class OllamaConfig:
    """Configuration for Ollama connection."""

    host: str = "http://192.168.3.16"
    model: str = "phi4"

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        """Create config from environment variables."""
        return cls(
            host=os.environ.get("OLLAMA_HOST", cls.host),
            model=os.environ.get("OLLAMA_MODEL", cls.model),
        )


class ControlServer:
    """Per-client control WebSocket server on port 9092.

    Handles end_audio events from the UI, triggers Ollama processing
    for each client's transcript, and streams results back.
    Supports multiple simultaneous clients via uid mapping.
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig.from_env()
        self.markdown_converter = MarkdownConverter()
        # uid -> websocket connection (active clients)
        self.clients: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def _get_ollama(self):
        """Create an OllamaProcessor instance, returns None if unavailable."""
        try:
            from ollama_processor import OllamaProcessor

            return OllamaProcessor(model=self.config.model, host=self.config.host)
        except Exception as e:
            logging.warning(f"Ollama unavailable: {e}")
            return None

    def _send(self, uid: str, data: Dict[str, Any]) -> bool:
        """Safely send JSON data to websocket. Returns True if successful."""
        with self.lock:
            websocket = self.clients.get(uid)
        if not websocket:
            logging.debug(f"[Control] Client {uid} not connected, skipping send")
            return False
        try:
            websocket.send(json.dumps(data))
            return True
        except Exception as e:
            logging.debug(f"[Control] Send failed for {uid}: {e}")
            return False

    def _process_transcript(self, uid: str, text: str) -> None:
        """Run Ollama processing for a client's transcript in a background thread."""
        logging.info(f"[Control] Processing transcript for {uid} ({len(text)} chars)")

        ollama = self._get_ollama()
        if not ollama:
            self._send(
                uid,
                {"type": "ollama_error", "uid": uid, "error": "Ollama not available"},
            )
            return

        try:
            # Send processing started notification
            if not self._send(uid, {"type": "ollama_start", "uid": uid}):
                return

            # Cleanup - convert markdown to HTML
            logging.info(f"[Control] Cleaning transcript for {uid}")
            cleaned = ollama.cleanup_transcript(text)
            if not self._send(
                uid,
                {
                    "type": "ollama_cleaned",
                    "uid": uid,
                    "text": cleaned,
                    "html": self.markdown_converter.convert(cleaned),
                },
            ):
                return

            # Summary - convert markdown to HTML
            logging.info(f"[Control] Summarizing for {uid}")
            summary = ollama.summarize(text)
            if not self._send(
                uid,
                {
                    "type": "ollama_summary",
                    "uid": uid,
                    "text": summary,
                    "html": self.markdown_converter.convert(summary),
                },
            ):
                return

            # Action items - convert markdown to HTML
            logging.info(f"[Control] Extracting actions for {uid}")
            actions = ollama.extract_action_items(text)
            actions_html = [self.markdown_converter.convert(item) for item in actions]
            if not self._send(
                uid,
                {
                    "type": "ollama_actions",
                    "uid": uid,
                    "items": actions,
                    "items_html": actions_html,
                },
            ):
                return

            # Done
            self._send(uid, {"type": "ollama_done", "uid": uid})
            logging.info(f"[Control] Ollama processing complete for {uid}")

        except Exception as e:
            logging.error(f"[Control] Ollama error for {uid}: {e}")
            self._send(websocket, {"type": "ollama_error", "uid": uid, "error": str(e)})

    def _handle_end_audio(self, message: Dict[str, Any], websocket) -> None:
        """Handle end_audio message from client."""
        uid = message.get("uid")
        text = message.get("transcript", "").strip()

        if not text:
            logging.info(f"[Control] end_audio from {uid} — no transcript text")
            self._send(
                uid,
                {
                    "type": "ollama_error",
                    "uid": uid,
                    "error": "No transcript text received",
                },
            )
            return

        # Process in background thread so we don't block other clients
        thread = threading.Thread(
            target=self._process_transcript, args=(uid, text), daemon=True
        )
        thread.start()

    def _handle_ping(self, uid: str, websocket) -> None:
        """Handle ping message from client."""
        self._send(websocket, {"type": "pong", "uid": uid})

    def handle_client(self, websocket) -> None:
        """Handle a single control channel client connection."""
        uid: Optional[str] = None
        logging.info("[Control] Client connected")

        try:
            for raw_message in websocket:
                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError:
                    continue

                msg_type = message.get("type")
                msg_uid = message.get("uid")

                # Register client uid on first message
                if msg_uid and uid is None:
                    uid = msg_uid
                    with self.lock:
                        self.clients[uid] = websocket
                    logging.info(f"[Control] Registered client {uid}")

                if msg_type == "ping":
                    self._handle_ping(uid, websocket)
                elif msg_type == "end_audio":
                    self._handle_end_audio(message, websocket)

        except Exception as e:
            logging.info(f"[Control] Client {uid} disconnected: {e}")
        finally:
            if uid:
                with self.lock:
                    self.clients.pop(uid, None)
            logging.info(f"[Control] Client {uid} cleaned up")

    def run(self, host: str, port: int, ssl_context=None) -> None:
        """Start the control WebSocket server."""
        protocol = "WSS" if ssl_context else "WS"
        logging.info(f"[Control] Starting {protocol} control server on {host}:{port}")
        with serve(self.handle_client, host, port, ssl_context=ssl_context) as server:
            server.serve_forever()


class TranscriptionHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for serving static files and proxying WebSocket."""

    static_dir = STATIC_DIR
    ws_port = 9090

    MIME_TYPES = {
        ".css": "text/css",
        ".js": "application/javascript",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
        ".html": "text/html; charset=utf-8",
        ".json": "application/json",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
        ".ttf": "font/ttf",
    }

    def _send_cors_headers(self) -> None:
        """Send CORS headers."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _serve_file(self, file_path: str) -> None:
        """Serve a file with appropriate content type."""
        try:
            with open(file_path, "rb") as f:
                content = f.read()

            # Determine content type from extension
            ext = os.path.splitext(file_path)[1].lower()
            content_type = self.MIME_TYPES.get(ext, "application/octet-stream")

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(content))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self._send_404()

    def _send_404(self) -> None:
        """Send 404 response."""
        self.send_response(404)
        self.send_header("Content-Type", "text/html")
        self._send_cors_headers()
        body = b"<h1>404 Not Found</h1><p>The requested resource was not found.</p>"
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Handle GET requests."""
        # Serve index.html for root or explicit path
        if self.path in ("/", "/index.html"):
            index_path = os.path.join(self.static_dir, "index.html")
            if os.path.exists(index_path):
                self._serve_file(index_path)
                return
            self._send_404()
            return

        # Serve other static files
        if self.path.startswith("/static/"):
            file_path = os.path.join(
                self.static_dir, self.path[8:]
            )  # Remove '/static/' prefix
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self._serve_file(file_path)
                return

        # Return 404 for other paths
        self._send_404()

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def log_message(self, format, *args) -> None:
        """Log HTTP requests."""
        logging.info(f"HTTP: {args[0]}")


def run_http_server(host: str, port: int, ssl_context=None) -> None:
    """Run HTTP server in a separate thread."""
    server = HTTPServer((host, port), TranscriptionHTTPHandler)
    if ssl_context:
        server.socket = ssl_context.wrap_socket(server.socket, server_side=True)
        protocol = "HTTPS"
    else:
        protocol = "HTTP"

    logging.info(f"{protocol} server started on {host}:{port}")
    logging.info(f"Serving static files from {STATIC_DIR}")
    server.serve_forever()


class PatchedServeClient:
    """Monkey-patch ServeClient.speech_to_text to handle disconnections gracefully."""

    @staticmethod
    def patch():
        """Apply the patch to ServeClient."""
        from whisper_live.server import ServeClient
        import json
        import logging
        import time

        original_speech_to_text = ServeClient.speech_to_text

        def patched_speech_to_text(self):
            """Patched version that handles closed connections gracefully."""
            while True:
                if self.exit:
                    logging.info("Exiting speech to text thread")
                    break

                if self.frames_np is None:
                    continue

                # clip audio if the current chunk exceeds 30 seconds
                if (
                    self.frames_np[
                        int((self.timestamp_offset - self.frames_offset) * self.RATE) :
                    ].shape[0]
                    > 25 * self.RATE
                ):
                    duration = self.frames_np.shape[0] / self.RATE
                    self.timestamp_offset = self.frames_offset + duration - 5

                samples_take = max(
                    0, (self.timestamp_offset - self.frames_offset) * self.RATE
                )
                input_bytes = self.frames_np[int(samples_take) :].copy()
                duration = input_bytes.shape[0] / self.RATE
                if duration < 1.0:
                    continue

                try:
                    input_sample = input_bytes.copy()

                    # whisper transcribe with prompt
                    result, info = self.transcriber.transcribe(
                        input_sample,
                        initial_prompt=self.initial_prompt,
                        language=self.language,
                        task=self.task,
                        vad_filter=True,
                        vad_parameters=self.vad_parameters,
                    )

                    if self.language is None:
                        if info.language_probability > 0.5:
                            self.language = info.language
                            logging.info(
                                f"Detected language {self.language} with probability {info.language_probability}"
                            )
                            try:
                                self.websocket.send(
                                    json.dumps(
                                        {
                                            "uid": self.client_uid,
                                            "language": self.language,
                                            "language_prob": info.language_probability,
                                        }
                                    )
                                )
                            except Exception:
                                self.exit = True
                                break
                        else:
                            continue

                    if len(result):
                        self.t_start = None
                        last_segment = self.update_segments(result, duration)
                        if len(self.transcript) < self.send_last_n_segments:
                            segments = self.transcript
                        else:
                            segments = self.transcript[-self.send_last_n_segments :]
                        if last_segment is not None:
                            segments = segments + [last_segment]
                    else:
                        # show previous output if there is pause
                        segments = []
                        if self.t_start is None:
                            self.t_start = time.time()
                        if time.time() - self.t_start < self.show_prev_out_thresh:
                            if len(self.transcript) < self.send_last_n_segments:
                                segments = self.transcript
                            else:
                                segments = self.transcript[-self.send_last_n_segments :]

                        # add a blank if there is no speech for 3 seconds
                        if len(self.text) and self.text[-1] != "":
                            if time.time() - self.t_start > self.add_pause_thresh:
                                self.text.append("")

                    try:
                        self.websocket.send(
                            json.dumps({"uid": self.client_uid, "segments": segments})
                        )
                    except Exception:
                        # Connection closed - exit gracefully
                        self.exit = True
                        break

                except Exception as e:
                    logging.error(f"[ERROR]: Failed to transcribe audio chunk: {e}")
                    time.sleep(0.01)

        ServeClient.speech_to_text = patched_speech_to_text


class TranscriptionServerWSS(TranscriptionServer):
    """Extended TranscriptionServer with WSS/SSL support."""

    def run(self, host: str, port: int = 9090, ssl_context=None) -> None:
        """
        Run the transcription server with optional WSS support.

        Args:
            host: The host address to bind the server.
            port: The port number to bind the server.
            ssl_context: SSL context for WSS.
        """
        # Apply the patch before running
        PatchedServeClient.patch()

        protocol = "WSS" if ssl_context else "WS"
        logging.info(f"Starting {protocol} transcription server on {host}:{port}")

        with serve(self.recv_audio, host, port, ssl_context=ssl_context) as server:
            server.serve_forever()


def create_ssl_context(cert_file: str, key_file: str) -> ssl.SSLContext:
    """Create SSL context from certificate and key files."""
    if not os.path.exists(cert_file):
        raise FileNotFoundError(f"Certificate file not found: {cert_file}")
    if not os.path.exists(key_file):
        raise FileNotFoundError(f"Key file not found: {key_file}")

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(cert_file, key_file)
    return ssl_context


def get_local_ips() -> set:
    """Get all local IP addresses for SAN."""
    local_ips = {"127.0.0.1", "::1"}
    try:
        hostname = socket.gethostname()
        local_ips.add(socket.gethostbyname(hostname))
    except Exception:
        pass

    # Add all network interface IPs
    try:
        import subprocess

        result = subprocess.run(["ifconfig"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if "inet " in line:
                parts = line.split()
                for part in parts:
                    if "." in part and part.count(".") == 3:
                        local_ips.add(part)
    except Exception:
        pass

    return local_ips


def generate_self_signed_cert(
    cert_file: str, key_file: str, host: str = "0.0.0.0"
) -> bool:
    """Generate a self-signed certificate for development with proper SAN."""
    import subprocess
    import tempfile

    logging.info("Generating self-signed certificate with SAN...")

    local_ips = get_local_ips()

    # Build SAN config
    ip_entries = "\n".join([f"IP.{i + 1} = {ip}" for i, ip in enumerate(local_ips)])

    config_content = f"""
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = {host if host != "0.0.0.0" else "localhost"}

[v3_req]
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
{ip_entries}
"""

    try:
        # Write config to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as f:
            f.write(config_content)
            config_file = f.name

        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:4096",
                "-nodes",
                "-out",
                cert_file,
                "-keyout",
                key_file,
                "-days",
                "365",
                "-config",
                config_file,
                "-extensions",
                "v3_req",
            ],
            check=True,
        )

        # Clean up temp file
        os.unlink(config_file)

        logging.info(f"Generated self-signed certificate: {cert_file}")
        logging.info(f"Certificate includes IPs: {', '.join(local_ips)}")
        logging.info("Note: You may need to trust this certificate in your browser")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to generate certificate: {e}")
        return False
    except FileNotFoundError:
        logging.error(
            "openssl not found. Please install OpenSSL (brew install openssl)"
        )
        return False


def trust_certificate(cert_file: str) -> bool:
    """Add certificate to system trust store (macOS)."""
    import subprocess

    logging.info("Adding certificate to system trust store...")
    try:
        subprocess.run(
            [
                "sudo",
                "security",
                "add-trusted-cert",
                "-d",
                "-r",
                "trustRoot",
                "-k",
                "/Library/Keychains/System.keychain",
                cert_file,
            ],
            check=True,
        )
        logging.info("Certificate added to system trust store successfully")
        logging.info("Please restart your browser for changes to take effect")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to trust certificate: {e}")
        logging.info("You can still manually trust the certificate in your browser")
        return False


def pre_download_models(model_sizes: Optional[list] = None) -> None:
    """Pre-download Whisper models to cache for faster startup.

    Args:
        model_sizes: List of model sizes to download. If None, downloads common models.
    """
    from faster_whisper.utils import download_model

    if model_sizes is None:
        model_sizes = ["base", "small", "medium"]

    logging.info("Pre-downloading models (this may take a while on first run)...")
    for size in model_sizes:
        try:
            logging.info(f"Downloading {size} model...")
            download_model(size)
            logging.info(f"  ✓ {size} model cached")
        except Exception as e:
            logging.warning(f"Failed to download {size}: {e}")
    logging.info("Model pre-download complete")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Whisper Live transcription server with web UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 server.py                       # Start with HTTPS (auto-generate cert if needed)
  python3 server.py --generate-cert       # Generate self-signed certificate
  python3 server.py --trust-cert          # Add certificate to system trust store (macOS)
  python3 server.py --http                # Start with HTTP only (insecure, dev only)
  python3 server.py --https-port 8443     # Custom HTTPS port
  python3 server.py --download-models     # Pre-download models and exit
  python3 server.py --model small         # Use specific model (tiny/base/small/medium/large)
        """,
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("WHISPER_HOST", "0.0.0.0"),
        help="Host to bind to (env: WHISPER_HOST)",
    )
    parser.add_argument(
        "--https-port",
        type=int,
        default=int(os.environ.get("HTTPS_PORT", 9090)),
        help="HTTPS port for web UI (env: HTTPS_PORT)",
    )
    parser.add_argument(
        "--wss-port",
        type=int,
        default=int(os.environ.get("WSS_PORT", 9091)),
        help="WSS port for audio (env: WSS_PORT)",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        default=os.environ.get("USE_HTTP", "").lower() in ("1", "true", "yes"),
        help="Use HTTP instead of HTTPS (env: USE_HTTP)",
    )
    parser.add_argument(
        "--generate-cert", action="store_true", help="Generate self-signed certificate"
    )
    parser.add_argument(
        "--trust-cert",
        action="store_true",
        help="Add certificate to system trust store (macOS)",
    )
    parser.add_argument(
        "--cert",
        type=str,
        default=os.environ.get("SSL_CERT", None),
        help="SSL certificate file (env: SSL_CERT)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=os.environ.get("SSL_KEY", None),
        help="SSL private key file (env: SSL_KEY)",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Pre-download models to cache and exit",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("WHISPER_MODEL", None),
        help="Specific model to pre-download (env: WHISPER_MODEL)",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=int(os.environ.get("CONTROL_PORT", 9092)),
        help="Control WebSocket port for Ollama processing (default: 9092, env: CONTROL_PORT)",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://192.168.3.16"),
        help="Ollama server URL (env: OLLAMA_HOST)",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_MODEL", "phi4"),
        help="Ollama model for post-processing (env: OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        default=os.environ.get("OLLAMA_DISABLED", "").lower() in ("1", "true", "yes"),
        help="Disable Ollama control server (env: OLLAMA_DISABLED)",
    )
    return parser.parse_args()


def setup_ssl(args) -> tuple:
    """Setup SSL context based on arguments."""
    cert_file = (
        args.cert if args.cert else os.path.join(os.path.dirname(__file__), "cert.pem")
    )
    key_file = (
        args.key if args.key else os.path.join(os.path.dirname(__file__), "key.pem")
    )

    # Handle --trust-cert separately
    if args.trust_cert:
        if os.path.exists(cert_file):
            trust_certificate(cert_file)
        else:
            logging.error(f"Certificate not found: {cert_file}")
            logging.info("Run 'python3 server.py --generate-cert' first")
        return None, cert_file, key_file

    # Generate certificate if requested or if files don't exist (and not using HTTP)
    need_cert = not args.http
    if args.generate_cert or (
        need_cert and (not os.path.exists(cert_file) or not os.path.exists(key_file))
    ):
        if not generate_self_signed_cert(cert_file, key_file, args.host):
            logging.error("Failed to generate certificate. Exiting.")
            return None, cert_file, key_file

    # Create SSL context (default to HTTPS unless --http specified)
    ssl_context = None
    use_https = not args.http
    if use_https:
        try:
            ssl_context = create_ssl_context(cert_file, key_file)
            logging.info("HTTPS/WSS enabled")
        except FileNotFoundError as e:
            logging.error(str(e))
            logging.error("Use --generate-cert to create a self-signed certificate")
            return None, cert_file, key_file
    else:
        logging.warning("Running in HTTP mode (insecure). Use only for development.")

    return ssl_context, cert_file, key_file


def main() -> None:
    args = parse_args()

    # Handle --download-models
    if args.download_models:
        models = [args.model] if args.model else None
        pre_download_models(models)
        return

    # Setup SSL
    result = setup_ssl(args)
    if result is None:
        return
    ssl_context, cert_file, key_file = result

    # Check if static files exist
    if not os.path.exists(STATIC_DIR):
        logging.error(f"Static directory not found: {STATIC_DIR}")
        logging.error("Please ensure the 'static' folder exists with index.html")
        return

    # Set static dir for HTTP handler
    TranscriptionHTTPHandler.static_dir = STATIC_DIR

    # Start HTTPS/HTTP server in background thread
    http_thread = threading.Thread(
        target=run_http_server,
        args=(args.host, args.https_port, ssl_context),
        daemon=True,
    )
    http_thread.start()

    # Start control server (Ollama processing) unless disabled
    if not args.no_ollama:
        config = OllamaConfig(host=args.ollama_host, model=args.ollama_model)
        control_server = ControlServer(config=config)
        control_thread = threading.Thread(
            target=control_server.run,
            args=(args.host, args.control_port, ssl_context),
            daemon=True,
        )
        control_thread.start()
        logging.info(f"Control server started on port {args.control_port}")
    else:
        logging.info("Ollama control server disabled (--no-ollama)")

    # Run WebSocket server in main thread
    ws_server = TranscriptionServerWSS()

    protocol = "WSS" if ssl_context else "WS"
    logging.info(f"Starting {protocol} server on {args.host}:{args.wss_port}")

    try:
        ws_server.run(args.host, args.wss_port, ssl_context=ssl_context)
    except KeyboardInterrupt:
        logging.info("Server stopped by user")


if __name__ == "__main__":
    main()
