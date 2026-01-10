#!/usr/bin/env python3
"""
Ollama Code Review Assistant - Professional Edition
"""

import asyncio
import re
import tempfile
import logging
import shutil
import os
import atexit
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, AsyncGenerator

import gradio as gr
import ollama

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://192.168.3.16")
DEFAULT_MODEL = "llama3.2"
MAX_FILE_SIZE_MB = 5  # Safety limit

# --- Business Logic: Ollama Service ---

class OllamaService:
    """Handles communications with the Ollama API with connection pooling."""

    def __init__(self, host: str):
        self.host = self._sanitize_host(host)
        self._client: Optional[ollama.AsyncClient] = None

    @property
    def client(self) -> ollama.AsyncClient:
        if self._client is None:
            self._client = ollama.AsyncClient(host=self.host)
        return self._client

    @staticmethod
    def _sanitize_host(host: str) -> str:
        host = host.strip().rstrip('/')
        if host and not host.startswith(('http://', 'https://')):
            return f"http://{host}"
        return host or "http://localhost:11434"

    async def get_models(self) -> List[str]:
        try:
            response = await self.client.list()
            return sorted([m.model for m in response.models])
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            return []

    async def generate_review(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        code_context: str
    ) -> AsyncGenerator[str, None]:
        full_user_content = f"# User Request\n{user_prompt}\n\n# Code Context\n{code_context}"

        try:
            stream = await self.client.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': full_user_content}
                ],
                stream=True,
                options=dict(num_ctx=16384, temperature=0.2) # Lower temp for code tasks
            )

            async for part in stream:
                yield part['message']['content']

        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield f"\n\n**Error**: {str(e)}"

# --- Business Logic: File Processing ---

class FileProcessor:
    """Handles reading source files and extracting refactored code."""

    def __init__(self):
        # Create a unique session directory
        self.session_id = str(uuid.uuid4())[:8]
        self.temp_dir = Path(tempfile.gettempdir()) / f"ollama_review_{self.session_id}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self.cleanup)

    def cleanup(self):
        """Removes temporary files on exit."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def read_files(self, file_objects: List) -> Tuple[str, List[str]]:
        if not file_objects:
            return "", []

        content_blocks = []
        filenames = []

        for file_obj in file_objects:
            path = Path(file_obj.name)

            # Security/Performance check
            if path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                content_blocks.append(f"### File: {path.name}\n*Skipped: File too large (> {MAX_FILE_SIZE_MB}MB)*")
                continue

            filenames.append(path.name)
            try:
                code = path.read_text(encoding='utf-8', errors='replace')
                lang = path.suffix[1:] or "text"
                content_blocks.append(f"### File: {path.name}\n```{lang}\n{code}\n```")
            except Exception as e:
                content_blocks.append(f"### File: {path.name}\n*Error reading file: {e}*")

        return "\n\n".join(content_blocks), filenames

    def extract_and_save_code(self, text: str, original_filenames: List[str]) -> List[str]:
        # Improved Regex: Handles ```python:path/to/file.py or ```python
        pattern = re.compile(r'```(?:\w+)?(?::?([\w\.\-\/]+))?\n(.*?)\n```', re.DOTALL)
        matches = pattern.findall(text)

        saved_paths = []
        for i, (hint, code) in enumerate(matches):
            if hint:
                fname = Path(hint).name
            elif i < len(original_filenames):
                fname = f"refactored_{original_filenames[i]}"
            else:
                fname = f"suggested_fix_{i+1}.py"

            try:
                out_path = self.temp_dir / fname
                out_path.write_text(code.strip(), encoding='utf-8')
                saved_paths.append(str(out_path))
            except Exception as e:
                logger.error(f"Save error: {e}")

        return saved_paths

# --- UI Components ---

class AppUI:
    def __init__(self):
        self.service = OllamaService(DEFAULT_HOST)
        self.processor = FileProcessor()
        self.is_cancelled = False

    async def handle_refresh(self, host: str):
        self.service = OllamaService(host)
        models = await self.service.get_models()
        if not models:
            return gr.update(choices=[], value=None), "⚠️ No models found. Check connection."
        return gr.update(choices=models, value=models[0] if models else None), "✅ Connected"

    async def handle_submit(self, host: str, model: str, sys_prompt: str, user_prompt: str, files: List):
        self.is_cancelled = False
        if not model or not files:
            yield "Please select a model and upload files.", None
            return

        self.service = OllamaService(host)
        code_context, original_names = self.processor.read_files(files)

        full_response = ""
        yield "🚀 Analyzing code...", None

        async for chunk in self.service.generate_review(model, sys_prompt, user_prompt, code_context):
            if self.is_cancelled:
                full_response += "\n\n🛑 *Stopped by user.*"
                yield full_response, None
                return

            full_response += chunk
            yield full_response, None

        # Auto-extract if code blocks exist
        if "```" in full_response:
            paths = self.processor.extract_and_save_code(full_response, original_names)
            if paths:
                yield full_response, paths
            else:
                yield full_response, None

    def cancel(self):
        self.is_cancelled = True

    def create_interface(self):
        with gr.Blocks(analytics_enabled=False, title="Ollama Code Review Assistant") as ui:
            gr.Markdown("# Ollama Code Review Assistant")

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Connection Settings", open=False):
                        host_input = gr.Textbox(label="Ollama Host", value=DEFAULT_HOST)
                        refresh_btn = gr.Button("Refresh Connection")
                        status_msg = gr.Markdown("Status: Unknown")

                    model_selector = gr.Dropdown(label="Model Selection", choices=[])

                    sys_prompt_input = gr.Textbox(
                        label="System Role",
                        value="You are an expert software engineer. Provide concise, actionable code reviews. Use markdown code blocks for refactored code.",
                        lines=2
                    )

                    user_prompt_input = gr.Textbox(
                        label="Instruction",
                        placeholder="e.g. Refactor for readability and fix bugs",
                        value="Review this code for security vulnerabilities and suggest optimizations."
                    )

                    file_input = gr.File(label="Upload Source Files", file_count="multiple")

                    with gr.Row():
                        submit_btn = gr.Button("Analyze", variant="primary")
                        cancel_btn = gr.Button("Stop")
                        clear_btn = gr.ClearButton()

                with gr.Column(scale=2):
                    output_markdown = gr.Markdown(label="Analysis", value="### Results will appear here...")
                    download_output = gr.File(label="Download Refactored Files")

            # Events
            clear_btn.add([user_prompt_input, file_input, output_markdown, download_output])

            ui.load(self.handle_refresh, [host_input], [model_selector, status_msg])
            refresh_btn.click(self.handle_refresh, [host_input], [model_selector, status_msg])

            submit_evt = submit_btn.click(
                self.handle_submit,
                inputs=[host_input, model_selector, sys_prompt_input, user_prompt_input, file_input],
                outputs=[output_markdown, download_output]
            )

            cancel_btn.click(self.cancel, None, None, cancels=[submit_evt])

        return ui

if __name__ == "__main__":
    app = AppUI()
    app.create_interface().launch(server_name="0.0.0.0",
                                  server_port=7860,
                                  theme=gr.themes.Default(primary_hue="blue"),
                                  pwa=True)
