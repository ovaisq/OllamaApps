#!/usr/bin/env python3
"""Ollama Code Review Assistant

A modern, production-grade Gradio interface for AI-powered code review using Ollama.
"""

import gradio as gr
from pathlib import Path
import ollama
import threading
import time
from typing import List, Optional, Tuple

# Thread-safe cancellation flag
cancel_flag = threading.Event()

# Default configuration
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"

# Supported file extensions for code review
CODE_EXTENSIONS = [
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".go", ".rs", ".swift", ".kt",
    ".php", ".sql", ".html", ".css", ".scss",
    ".yml", ".yaml", ".json", ".xml",
    ".sh", ".bash", ".ps1", ".r", ".m"
]

def get_available_models(host: str) -> List[str]:
    """Fetch available models from the Ollama server."""
    if not host or not host.strip():
        return []
    try:
        client = ollama.Client(host=host.strip())
        response = client.list()
        models = [model.model for model in response.models]
        return sorted(models) if models else []
    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        return []

def refresh_models_simple(host: str):
    """Refresh the model selector with latest available models."""
    models = get_available_models(host)
    
    if not models:
        choices = []
        value = None
        status = f"**Warning**: No models found at {host}."
    else:
        choices = models
        value = models[0]
        status = f"**Success**: Found {len(models)} model(s)."
    
    # Use gr.update to preserve the elem_id and other configurations
    return gr.update(choices=choices, value=value), status

def read_code_files(file_objects: List) -> Tuple[str, Optional[str]]:
    """Read and combine multiple code files."""
    if not file_objects:
        return "", "No files uploaded"
    
    files = file_objects if isinstance(file_objects, list) else [file_objects]
    all_code_content = []
    
    for file_obj in files:
        if cancel_flag.is_set():
            return "", "Processing cancelled by user"
        try:
            file_path = Path(file_obj.name)
            code_content = file_path.read_text(encoding='utf-8')
            all_code_content.append(
                f"### File: {file_path.name}\n```{file_path.suffix[1:]}\n{code_content}\n```"
            )
        except Exception as e:
            all_code_content.append(f"### File: {file_path.name}\n*Error reading file: {str(e)}*")
    
    return "\n\n".join(all_code_content), None

def call_ollama_api(client: ollama.Client, model: str, prompt: str) -> dict:
    """Make API call to Ollama in a separate thread."""
    result = {}
    try:
        response = client.chat(
            model=model,
            options=dict(num_ctx=8192, temperature=0.7),
            messages=[{"role": "user", "content": prompt}]
        )
        result["response"] = response["message"]["content"]
    except Exception as e:
        result["error"] = str(e)
    return result

def process_code(host: str, selected_model: str, user_prompt: str, file_upload) -> tuple:
    """Process uploaded code files with AI analysis."""
    cancel_flag.clear()
    
    if not all([host, selected_model, user_prompt, file_upload]):
        return "**Error**: Missing required inputs", None
    
    combined_code, error = read_code_files(file_upload)
    if error: return f"**Error**: {error}", None
    
    refactor_request = any(kw in user_prompt.lower() for kw in ['refactor', 'rewrite', 'improve', 'fix'])
    
    full_prompt = f"# Request\n{user_prompt}\n\n# Files\n{combined_code}"
    
    try:
        client = ollama.Client(host=host.strip())
    except Exception as e:
        return f"**Connection Error**: {str(e)}", None
    
    response_data = {}
    def make_request():
        response_data.update(call_ollama_api(client, selected_model, full_prompt))
    
    api_thread = threading.Thread(target=make_request, daemon=True)
    api_thread.start()
    
    while api_thread.is_alive():
        if cancel_flag.is_set(): return "**Cancelled**", None
        time.sleep(0.1)
    
    if "response" in response_data:
        ai_res = response_data['response']
        refactored = extract_code_blocks(ai_res, file_upload) if refactor_request else []
        return f"## AI Analysis\n\n{ai_res}", refactored if refactored else None
    
    return f"**Error**: {response_data.get('error', 'Unknown error')}", None

def cancel_processing() -> str:
    cancel_flag.set()
    return "**Cancellation requested**..."

def extract_code_blocks(response_text: str, original_files: List) -> List[str]:
    import re
    import tempfile
    output_dir = Path(tempfile.gettempdir()) / "ollama_refactored"
    output_dir.mkdir(exist_ok=True)
    
    matches = re.findall(r'```([\w\.\-\/]*)\n(.*?)```', response_text, re.DOTALL)
    if not matches: return []
    
    files = original_files if isinstance(original_files, list) else [original_files]
    saved_files = []
    
    for idx, (ident, code) in enumerate(matches):
        fname = ident if '.' in ident else f"refactored_{Path(files[0].name).name}"
        out_path = output_dir / Path(fname).name
        out_path.write_text(code.strip(), encoding='utf-8')
        saved_files.append(str(out_path))
    
    return saved_files

# --- UPDATED CUSTOM CSS ---
CUSTOM_CSS = """
:root {
    --primary-bg: #0a0e1a;
    --glass-bg: rgba(20, 25, 45, 0.8);
    --glass-border: rgba(255, 255, 255, 0.08);
    --accent-primary: #6366f1;
    --text-primary: #f1f5f9;
    --code-bg: rgba(15, 23, 42, 0.95);
    --radius-sm: 8px;
}

body, .gradio-container {
    background: var(--primary-bg) !important;
    color: var(--text-primary) !important;
}

/* Scrollable Model Selector Styling */
#model-selector-container {
    max-height: 105px !important; /* Height for ~2 lines */
    overflow-y: auto !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-sm) !important;
    background: var(--code-bg) !important;
    padding: 8px !important;
}

/* Force 1 model per line */
#model-selector-container .gr-radio-group, 
#model-selector-container div.gap {
    display: flex !important;
    flex-direction: column !important;
    flex-wrap: nowrap !important;
}

/* Ensure labels take full width and look like list items */
#model-selector-container label {
    width: 100% !important;
    margin-bottom: 4px !important;
    background: rgba(255, 255, 255, 0.03) !important;
}

/* Standard Gradio block styling */
.gradio-container .block {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
}

#results-box {
    background: var(--code-bg) !important;
    padding: 1.5rem !important;
    min-height: 400px !important;
    overflow-y: auto !important;
}
footer {display: none !important;}
"""

def create_interface():
    with gr.Blocks(title="Ollama Code Review Assistant", css=CUSTOM_CSS) as ui:
        gr.Markdown("# Ollama Code Review Assistant")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Configuration")
                with gr.Row():
                    host = gr.Textbox(label="Ollama Host", value=DEFAULT_HOST, scale=3)
                    refresh_btn = gr.Button("Refresh", scale=1)
                
                status_msg = gr.Markdown(value="")
                
                # --- UPDATED MODEL SELECTOR ---
                initial_models = get_available_models(DEFAULT_HOST)
                model_selector = gr.Radio(
                    label="Select Model (Scrollable)",
                    choices=initial_models,
                    value=initial_models[0] if initial_models else None,
                    interactive=True,
                    elem_id="model-selector-container" # ID for CSS targeting
                )
                
                prompt = gr.Textbox(label="Analysis Prompt", placeholder="Review for bugs...", lines=4)
                file_upload = gr.File(label="Upload Code", file_count="multiple")
                
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary", scale=2)
                    cancel_btn = gr.Button("Cancel", variant="secondary", scale=1)
            
            with gr.Column(scale=3):
                output = gr.Markdown(value="*Results...*", elem_id="results-box")
                download_files = gr.File(label="Refactored Files", interactive=False)
        
        refresh_btn.click(refresh_models_simple, [host], [model_selector, status_msg])
        submit_btn.click(process_code, [host, model_selector, prompt, file_upload], [output, download_files])
        cancel_btn.click(cancel_processing, None, [output])
    
    return ui

if __name__ == "__main__":
    create_interface().launch(server_name='0.0.0.0', server_port=7860)
