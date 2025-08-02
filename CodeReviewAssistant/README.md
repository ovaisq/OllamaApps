# Ollama Code Review Assistant

A Gradio-based web interface for code review and analysis using local language models via Ollama.

## Features

- **Multi-language Support**: Analyzes code in Python, JavaScript, Java, C/C++, Ruby, Go, TypeScript, PHP, SQL, HTML, CSS, YAML, and more.
- **Local LLM Integration**: Connects to an Ollama server for on-device processing.
- **Multiple File Uploads**: Process multiple files simultaneously.
- **Model Selection**: Choose from available local models.
- **Custom Prompts**: Provide specific instructions for code analysis.
- **Cancellation Support**: Stop in-progress requests with a cancel button.
- **User-Friendly Interface**: Clean, intuitive web interface.

## Requirements

- Python 3.7+
- Gradio
- Ollama client library (`ollama`)

## Installation

1. Install dependencies:

    ```bash
    pip install gradio ollama
    ```

2. Ensure the Ollama server is running locally or accessible at the specified host.

## Usage

1. Run the application:

    ```bash
    python ui.py
    ```

2. Access the web interface at `http://localhost:7860`.

3. Configure the Ollama host address (default: `http://192.168.3.16`).

4. Upload one or more code files using the file uploader.

5. Select an available LLM model from the dropdown menu.

6. Enter your analysis prompt in the text box.

7. Click "Submit" to get the response from the LLM.

8. Use the "Cancel Processing" button to stop in-progress requests.

## Configuration

### Ollama Host

- Default: `http://192.168.3.16`
- Change this to your Ollama server address if different.

### Supported File Types

- `.py` (Python)
- `.js` (JavaScript)
- `.java` (Java)
- `.c`, `.cpp` (C/C++)
- `.rb` (Ruby)
- `.go` (Go)
- `.ts` (TypeScript)
- `.php` (PHP)
- `.sql` (SQL)
- `.html`, `.css` (HTML/CSS)
- `.yml`, `.yaml` (YAML)
- `.jsx`, `.tsx` (React JSX/TSX)

## Example Prompts
- Please analyze this code for potential security vulnerabilities.
- Review the code structure and suggest improvements.
- Identify any performance bottlenecks in this implementation.

## Dependencies

- `gradio`: For the web interface
- `ollama`: For communication with local LLMs
- `pathlib`: For file path handling
- `threading`, `time`: For asynchronous processing and cancellation support

## Security Notes

- All code processing occurs locally on your machine.
- No code is sent to external servers.
- Ensure your Ollama server is properly secured if accessible over a network.

## Troubleshooting

### Connection Issues

- Verify the Ollama server is running: `ollama serve`.
- Check the host address in the interface.
- Ensure the firewall allows connections to the specified port.

### Model Not Found

- Make sure the selected model is downloaded locally.
- Use `ollama list` to see available models.
- Download required models using `ollama pull <model-name>`.

## Development

The application uses a threaded approach for API calls to support cancellation functionality. The interface supports real-time model updates when changing the host address.

## Author
Ollama QWEN3-CODER
Code Review Assistant Script
For issues or feature requests, please create an issue on the repository.
