#!/usr/bin/env python3
"""Code Review Assistant Script
        This script provides a Gradio-based web interface to upload code files,
    specify a prompt, select an Ollama model, and receive analysis from a local
    language model (LLM) via Ollama.
        Features:
         - Supports multiple programming languages through file type filtering.
         - Connects to an Ollama server for LLM processing.
         - Provides user-friendly error handling and feedback.
         - Allows selection of available models from the Ollama server.
        Usage:
         1. Set the Ollama server address in the "Ollama Host" input field.
         2. Upload a code file using the file uploader.
         3. Select an LLM model from the dropdown menu.
         4. Enter your prompt in the "Prompt" text box to guide the analysis.
         5. Click on the "Submit" button to get the response from the LLM.
        Dependencies:
         - gradio
         - ollama
"""

import gradio as gr
from pathlib import Path
import ollama

def get_available_models(host):
    """Fetch available models from the Ollama server.
        Args:
            host (str): The URL of the Ollama server.
        Returns:
            list: List of model names.
    """

    try:
        client = ollama.Client(host=host)
        response = client.list()
        # Extract model names from the response
        return [model.model for model in response.models]
    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        return []

def process_code(host, selected_model, user_prompt, file_upload):
    """Process uploaded code file with a custom prompt using Ollama.

        Args:
            host (str): The URL of the Ollama server.
            selected_model (str): The name of the selected model.
            user_prompt (str): The prompt to guide the LLM's analysis.
            file_upload (gr.File): Uploaded code file object.

        Returns:
            str: The response from the LLM or an error message.
    """

    if file_upload is None:
        return "Please upload a code file"
    try:
        # Read uploaded file
        code_content = Path(file_upload.name).read_text()
        # Construct the full prompt
        full_prompt = f"{user_prompt}\n\n{code_content}"
        # Initialize Ollama client with specified host
        client = ollama.Client(host=host)
        # Get model response
        response = client.chat(
            model=selected_model,
            options=dict(num_ctx=4096),
            messages=[{
                "role": "user",
                "content": full_prompt
            }]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# Custom CSS for a lighter color theme
custom_css = """
body {
    background-color: #f8f9fa;
    font-family: Arial, sans-serif;
}
.markdown {
    background-color: rgba(255, 255, 255, 0.7);
    border-radius: 5px;
    padding: 10px;
}
#results-box {
    border: 1px solid #ccc;
    padding: 10px;
    border-radius: 5px;
    background-color: rgba(0, 0, 0, 0.9);
    min-height: 100px;
}
.gradio-container {
    display: flex;
    max-width: 1200px;
    margin: auto;
}
.inputs-column {
    width: 30%;
    padding-right: 20px;
}
.outputs-column {
    width: 70%;
}
button {
    background-color: #4BA3C7; /* Sky Blue */
    border: none;
    color: white;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 5px;
}
button:hover {
    background-color: #45a049;
}
.gr-file-wrapper .close-button svg {
    width: 16px;
        height: 16px;
        }
footer {display: none !important;}
"""

def update_models(host):
    """Fetch and return available models based on the provided host."""

    return gr.Dropdown.update(choices=get_available_models(host))

# Create Gradio interface
with gr.Blocks(title="Ollama Code Review Assistant", css=custom_css) as ui:
    gr.Markdown("# Ollama Code Review Assistant")
    with gr.Row():
        with gr.Column(elem_id="inputs-column", scale=1):
            host = gr.Textbox(label="Ollama Host", value="http://")
            model_dropdown = gr.Dropdown(
                label="Select Model",
                choices=get_available_models("http://"),
                value="devstral", allow_custom_value=True
            )
            host.change(fn=update_models, inputs=host, outputs=model_dropdown)
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Ask about code structure, potential issues, etc."
            )
            file_upload = gr.File(
                label="Upload Code File",
                file_types=[".py", ".js", ".java", ".c", ".cpp", ".rb", ".go", ".ts", ".php", ".sql", ".html", ".css",".yml",".yaml",".jsx",".tsx"]
            )
            submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column(elem_id="outputs-column", scale=3):
            output = gr.Markdown(label="Model Response", elem_id="results-box")

    submit_btn.click(
        fn=process_code,
        inputs=[host, model_dropdown, prompt, file_upload],
        outputs=output
    )

if __name__ == "__main__":
    ui.launch(server_name='0.0.0.0', server_port=7860, pwa=True)
