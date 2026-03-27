#!/usr/bin/env python3
"""Ollama integration for processing transcribed text using the official Ollama SDK.

This module provides post-processing capabilities for Whisper transcriptions:
- Text cleanup and formatting
- Summarization
- Translation
- Q&A on transcript content

Configuration via .env file or environment variables:
  OLLAMA_HOST    Ollama server URL (default: http://192.168.3.16)
  OLLAMA_MODEL   Model to use     (default: phi4)
  OLLAMA_TIMEOUT Request timeout in seconds (default: 300)
"""

import argparse
import logging
import os
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Callable, List
from datetime import datetime


# Load .env file if present (walk up from this file's location)
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

# Ollama defaults from environment
_DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.3.16")
_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "phi4")
_DEFAULT_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", 300))

try:
    import ollama
except ImportError:
    ollama = None
    logging.warning("ollama package not installed. Run: pip install ollama")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OllamaProcessor:
    """Process transcribed text using Ollama models via the official SDK."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        """Initialize Ollama processor.

        Args:
            model: Ollama model name to use (e.g., llama3.2, mistral, qwen2.5)
            host: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

        # Initialize client with custom host
        if ollama:
            self.client = ollama.Client(host=host, timeout=timeout)
        else:
            self.client = None

        # Verify Ollama is running
        self._check_connection()

    def _check_connection(self):
        """Check if Ollama server is accessible."""
        try:
            response = self.client.list()
            # Ollama SDK returns a ListResponse object with .models attribute
            models = response.models if hasattr(response, "models") else []
            if models:
                logger.info(f"Connected to Ollama at {self.host}")
                # Model objects have 'model' attribute (not 'name')
                model_names = [getattr(m, "model", str(m)) for m in models[:5]]
                logger.info(f"Available models: {', '.join(model_names)}")
            else:
                logger.warning(f"Ollama returned empty model list")
        except Exception as e:
            logger.warning(f"Cannot connect to Ollama at {self.host}: {e}")
            logger.warning(
                "Continuing without Ollama - enhancement features will be disabled"
            )

    def _generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
    ):
        """Generate text using Ollama via the official SDK.

        Args:
            prompt: User prompt
            system: System prompt
            stream: Whether to stream the response

        Returns:
            Generated text or generator for streaming
        """
        try:
            if stream:
                return self._stream_generate(prompt, system)
            else:
                response = self.client.generate(
                    model=self.model, prompt=prompt, system=system, stream=False
                )
                return response["response"], response.get("context", [])
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    def _stream_generate(self, prompt: str, system: Optional[str] = None):
        """Stream generation using the official SDK."""
        stream = self.client.generate(
            model=self.model, prompt=prompt, system=system, stream=True
        )
        for chunk in stream:
            if chunk.get("done", False):
                break
            if "response" in chunk:
                yield chunk["response"]

    def cleanup_transcript(self, text: str) -> str:
        """Clean up and format transcribed text.

        Fixes punctuation, capitalization, and removes artifacts.
        """
        system = """You are a transcription cleanup assistant. Your job is to:
1. Fix capitalization (capitalize first letter of sentences, proper nouns)
2. Add proper punctuation (periods, commas, question marks)
3. Remove repeated words or phrases (e.g., "random stuff. random stuff" -> "random stuff")
4. Remove filler words (um, uh, like) when they don't add meaning
5. Keep the EXACT same words and meaning - DO NOT rephrase or change words
6. Do NOT hallucinate or add content that wasn't in the original
7. Format as clean, readable text

IMPORTANT: Only fix formatting issues. Never change "random" to "rendering" or any other word changes.

Return only the cleaned text, no explanations."""

        prompt = f"""Clean up this transcribed text by fixing only capitalization, punctuation, and removing repeated words. Keep all original words exactly the same:

{text}

Cleaned text:"""

        cleaned, _ = self._generate(prompt, system=system)
        return cleaned.strip()

    def summarize(self, text: str, length: str = "medium") -> str:
        """Summarize transcribed text.

        Args:
            text: Text to summarize
            length: Summary length (short/medium/long)
        """
        length_instructions = {
            "short": "1-2 sentences",
            "medium": "one paragraph (3-5 sentences)",
            "long": "2-3 paragraphs with key points",
        }

        system = f"""You are a summarization assistant. Create a clear, concise summary.
Length: {length_instructions.get(length, "one paragraph")}

Focus on:
- Main ideas and key points
- Important details and conclusions
- Action items if mentioned

Return only the summary, no explanations."""

        prompt = f"""Please summarize this text:

{text}"""

        summary, _ = self._generate(prompt, system=system)
        return summary

    def translate(self, text: str, target_language: str = "Spanish") -> str:
        """Translate text to another language.

        Args:
            text: Text to translate
            target_language: Target language
        """
        system = f"""You are a professional translator. Translate the following text to {target_language}.

Guidelines:
- Preserve the original meaning and tone
- Use natural, idiomatic {target_language}
- Maintain any technical terms appropriately
- Return only the translation, no explanations"""

        prompt = f"""Translate this text to {target_language}:

{text}"""

        translated, _ = self._generate(prompt, system=system)
        return translated

    def extract_action_items(self, text: str) -> List[str]:
        """Extract action items and tasks from text."""
        system = """Extract all action items, tasks, and commitments from the text.
Format as a bulleted list.
If no action items are found, return "No action items identified."
Return only the list, no explanations."""

        prompt = f"""Extract action items from this text:

{text}"""

        result, _ = self._generate(prompt, system=system)
        # Parse bullet points
        items = [
            line.strip().lstrip("-•*").strip()
            for line in result.split("\n")
            if line.strip()
            and (line.strip()[0] in "-•*" or line.strip().startswith("No action"))
        ]
        return items if items else [result.strip()]

    def answer_questions(self, text: str, question: str) -> str:
        """Answer questions about the transcribed text.

        Args:
            text: Source text
            question: Question to answer
        """
        system = """You are a Q&A assistant. Answer questions based ONLY on the provided text.
If the answer cannot be found in the text, say "I cannot find that information in the transcript."
Be concise and accurate."""

        prompt = f"""Based on this text:

{text}

Answer this question: {question}"""

        answer, _ = self._generate(prompt, system=system)
        return answer

    def chat(self, text: str, interactive: bool = True):
        """Interactive chat about the transcribed text.

        Args:
            text: Source text to discuss
            interactive: If True, run interactive mode. If False, return function for API use.
        """
        print("\n" + "=" * 60)
        print("Ollama Transcript Chat")
        print("=" * 60)
        print(f"Model: {self.model}")
        print("Commands: /summarize, /cleanup, /translate, /actions, /quit")
        print("=" * 60 + "\n")

        context = None

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "/quit":
                    print("Goodbye!")
                    break

                elif user_input.lower() == "/summarize":
                    print("\nSummarizing...")
                    result = self.summarize(text)
                    print(f"\nSummary: {result}\n")

                elif user_input.lower() == "/cleanup":
                    print("\nCleaning up transcript...")
                    result = self.cleanup_transcript(text)
                    print(f"\nCleaned: {result}\n")

                elif user_input.lower() == "/translate":
                    lang = (
                        input("Target language (default: Spanish): ").strip()
                        or "Spanish"
                    )
                    print(f"\nTranslating to {lang}...")
                    result = self.translate(text, lang)
                    print(f"\nTranslation: {result}\n")

                elif user_input.lower() == "/actions":
                    print("\nExtracting action items...")
                    items = self.extract_action_items(text)
                    print("\nAction Items:")
                    for item in items:
                        print(f"  • {item}")
                    print()

                else:
                    # Answer question about the text
                    print("\nThinking...")
                    system = """You are a helpful assistant discussing a transcript.
Be concise and accurate. Base your answers on the transcript content."""

                    prompt = f"""Transcript:
{text}

User question: {user_input}"""

                    response, _ = self._generate(prompt, system=system)
                    print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")


class StreamingOllamaProcessor:
    """Real-time streaming processor for live transcription enhancement.

    Buffers transcription segments and processes them in the background.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
        buffer_size: int = 500,
        processing_delay: float = 2.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.buffer_size = buffer_size
        self.processing_delay = processing_delay

        self.buffer = ""
        self.lock = threading.Lock()
        self.queue = queue.Queue()
        self.running = False
        self.thread = None
        self.callbacks: List[Callable[[str], None]] = []

    def add_transcript(self, text: str):
        """Add new transcribed text to the buffer."""
        with self.lock:
            self.buffer += " " + text
            # Could trigger background processing here
            if len(self.buffer) >= self.buffer_size:
                self._process_buffer()

    def _process_buffer(self):
        """Process the current buffer content."""
        if not self.buffer.strip():
            return

        # Simple cleanup without Ollama (can be extended)
        cleaned = self.buffer.strip()
        for callback in self.callbacks:
            try:
                callback(cleaned)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def register_callback(self, callback: Callable[[str], None]):
        """Register a callback for processed text."""
        self.callbacks.append(callback)

    def start(self):
        """Start background processing."""
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop background processing."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _process_loop(self):
        """Background processing loop."""
        while self.running:
            time.sleep(self.processing_delay)
            with self.lock:
                if self.buffer.strip():
                    self._process_buffer()


def list_ollama_models(host: str = _DEFAULT_HOST) -> List[dict]:
    """List available Ollama models using the official SDK."""
    if not ollama:
        logger.error("ollama package not installed")
        return []

    try:
        client = ollama.Client(host=host)
        result = client.list()
        return result.get("models", [])
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
    return []


def main():
    if not ollama:
        logger.error("Please install the ollama package: pip install ollama")
        return

    parser = argparse.ArgumentParser(
        description="Process transcribed text with Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 ollama_processor.py --model llama3.2
  python3 ollama_processor.py --file transcript.txt --summarize
  python3 ollama_processor.py --model mistral --interactive

Available models (run 'ollama list' to see installed):
  - llama3.2: Lightweight, fast
  - mistral: Good balance
  - qwen2.5: Strong reasoning
  - codellama: Code-focused
        """,
    )
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Ollama model to use")
    parser.add_argument("--host", default=_DEFAULT_HOST, help="Ollama server URL")
    parser.add_argument("--file", type=str, help="Input transcript file")
    parser.add_argument("--summarize", action="store_true", help="Generate summary")
    parser.add_argument("--cleanup", action="store_true", help="Clean up transcript")
    parser.add_argument("--translate", type=str, help="Translate to language")
    parser.add_argument("--actions", action="store_true", help="Extract action items")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive chat mode"
    )

    args = parser.parse_args()

    # Create processor (this will check connection)
    try:
        processor = OllamaProcessor(model=args.model, host=args.host)
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        return

    # Load transcript
    if args.file:
        with open(args.file, "r") as f:
            text = f.read()
    else:
        text = input("Paste transcript (end with empty line):\n")
        while True:
            line = input()
            if not line:
                break
            text += "\n" + line

    if not text.strip():
        print("No text provided. Exiting.")
        return

    # Execute requested operations
    if args.summarize:
        print("\n" + "=" * 40)
        print("SUMMARY")
        print("=" * 40)
        print(processor.summarize(text))

    if args.cleanup:
        print("\n" + "=" * 40)
        print("CLEANED TRANSCRIPT")
        print("=" * 40)
        print(processor.cleanup_transcript(text))

    if args.translate:
        print("\n" + "=" * 40)
        print(f"TRANSLATION ({args.translate})")
        print("=" * 40)
        print(processor.translate(text, args.translate))

    if args.actions:
        print("\n" + "=" * 40)
        print("ACTION ITEMS")
        print("=" * 40)
        items = processor.extract_action_items(text)
        for item in items:
            print(f"  • {item}")

    if args.interactive or not (
        args.summarize or args.cleanup or args.translate or args.actions
    ):
        processor.chat(text, interactive=True)


if __name__ == "__main__":
    main()
