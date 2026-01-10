#!/usr/bin/env python3
"""
Ollama Code Review Assistant
"""

import asyncio
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, AsyncGenerator

import gradio as gr
import ollama

# --- Configuration & Constants ---
DEFAULT_HOST = "http://192.168.3.16"
DEFAULT_MODEL = "llama3.2"
TEMP_DIR_PREFIX = "ollama_review_"

# --- Business Logic: Ollama Service ---

class OllamaService:
    """Handles communications with the Ollama API."""

    def __init__(self, host: str):
        # Clean host string to prevent protocol/formatting errors
        host = host.strip().rstrip('/')
        if host and not host.startswith(('http://', 'https://')):
            host = f"http://{host}"
        self.host = host or DEFAULT_HOST

    async def get_models(self) -> List[str]:
        """Fetches available models using a fresh client instance."""
        try:
            client = ollama.AsyncClient(host=self.host)
            response = await client.list()
            return sorted([m.model for m in response.models])
        except Exception as e:
            print(f"Model fetch error: {e}")
            return []

    async def generate_review(
        self,
        model: str,
        prompt: str,
        code_context: str
    ) -> AsyncGenerator[str, None]:
        """Streams the AI response using a fresh client instance."""
        full_prompt = f"# Request\n{prompt}\n\n# Files\n{code_context}"

        try:
            client = ollama.AsyncClient(host=self.host)
            message = {'role': 'user', 'content': full_prompt}
            async for part in await client.chat(
                model=model,
                messages=[message],
                stream=True,
                options=dict(num_ctx=8192, temperature=0.7)
            ):
                yield part['message']['content']
        except Exception as e:
            yield f"**API Error**: {str(e)}"

# --- Business Logic: File Processing ---

class FileProcessor:
    """Handles reading source files and extracting refactored code."""

    @staticmethod
    def read_files(file_objects: List) -> Tuple[str, List[str]]:
        if not file_objects:
            return "", []

        content_blocks = []
        file_list = file_objects if isinstance(file_objects, list) else [file_objects]

        for file_obj in file_list:
            path = Path(file_obj.name)
            try:
                code = path.read_text(encoding='utf-8', errors='replace')
                lang = path.suffix[1:] or "text"
                content_blocks.append(f"### File: {path.name}\n```{lang}\n{code}\n```")
            except Exception as e:
                content_blocks.append(f"### File: {path.name}\n*Error reading file: {e}*")

        return "\n\n".join(content_blocks), [Path(f.name).name for f in file_list]

    @staticmethod
    def extract_and_save_code(text: str, original_filenames: List[str]) -> List[str]:
        """Extracts code blocks and saves them to a temp directory for download."""
        output_dir = Path(tempfile.gettempdir()) / "ollama_refactored"
        output_dir.mkdir(exist_ok=True)

        # Pattern to find code blocks with optional filenames/languages
        pattern = re.compile(r'```(?:\w+)?(?::([\w\.\-\/]+))?\n(.*?)\n```', re.DOTALL)
        matches = pattern.findall(text)

        saved_paths = []
        for i, (filename_hint, code) in enumerate(matches):
            # Determine filename: 1. Hint in markdown 2. Original filename 3. Default
            if filename_hint:
                fname = Path(filename_hint).name
            elif i < len(original_filenames):
                fname = f"refactored_{original_filenames[i]}"
            else:
                fname = f"refactored_block_{i+1}.txt"

            out_path = output_dir / fname
            out_path.write_text(code.strip(), encoding='utf-8')
            saved_paths.append(str(out_path))

        return saved_paths

# --- UI Components & Styling ---

CUSTOM_CSS = """
.container { max-width: 1200px; margin: auto; }
.model-box {
    max-height: 150px;
    overflow-y: auto;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 5px;
}
.results-area {
    min-height: 500px;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 15px;
}
footer { display: none !important; }
"""

class AppUI:
    def __init__(self):
        self.service = OllamaService(DEFAULT_HOST)
        self.is_cancelled = False

    async def handle_refresh(self, host: str):
        self.service = OllamaService(host)
        models = await self.service.get_models()
        if not models:
            return gr.update(choices=[], value=None), "No models found. Check URL."
        return gr.update(choices=models, value=models[0]), f"Found {len(models)} models."

    async def handle_submit(self, host: str, model: str, prompt: str, files: List):
        self.is_cancelled = False
        if not model or not files:
            yield "Please select a model and upload files.", None
            return

        # 1. Prepare files
        code_context, original_names = FileProcessor.read_files(files)

        # 2. Stream AI Response
        full_response = ""
        async for chunk in self.service.generate_review(model, prompt, code_context):
            if self.is_cancelled:
                full_response += "\n\n *Process cancelled by user.*"
                yield full_response, None
                return

            full_response += chunk
            yield full_response, None

        # 3. Post-process refactored code if requested
        if any(k in prompt.lower() for k in ['refactor', 'fix', 'rewrite', 'improve']):
            refactored_paths = FileProcessor.extract_and_save_code(full_response, original_names)
            yield full_response, refactored_paths
        else:
            yield full_response, None

    def cancel(self):
        self.is_cancelled = True

    def create_interface(self):
        with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
            gr.Markdown("# Ollama Code Review Assistant")

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### Settings")
                        host_input = gr.Textbox(label="Ollama Host", value=DEFAULT_HOST)
                        refresh_btn = gr.Button("Refresh Models", size="sm")
                        status_msg = gr.Markdown("")

                        model_selector = gr.Radio(
                            label="Available Models",
                            choices=[],
                            elem_classes="model-box"
                        )

                    gr.Markdown("### Request")
                    prompt_input = gr.Textbox(
                        label="Instructions",
                        placeholder="e.g. Find security vulnerabilities and refactor for performance...",
                        lines=4
                    )
                    file_input = gr.File(label="Source Files", file_count="multiple")

                    with gr.Row():
                        submit_btn = gr.Button("Analyze", variant="primary")
                        cancel_btn = gr.Button("Stop")

                with gr.Column(scale=2):
                    gr.Markdown("### Analysis Result")
                    output_markdown = gr.Markdown(
                        "Upload files and click Analyze to start...",
                        elem_classes="results-area"
                    )
                    download_output = gr.File(label="Download Refactored Code")

            # Events
            demo.load(self.handle_refresh, [host_input], [model_selector, status_msg])
            refresh_btn.click(self.handle_refresh, [host_input], [model_selector, status_msg])

            submit_event = submit_btn.click(
                self.handle_submit,
                [host_input, model_selector, prompt_input, file_input],
                [output_markdown, download_output]
            )

            cancel_btn.click(self.cancel, cancels=[submit_event])

        return demo

if __name__ == "__main__":
    app = AppUI()
    app.create_interface().launch(server_name='0.0.0.0', server_port=7860)
