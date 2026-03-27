# GitHub PR Code Reviewer using Ollama

An automated code review tool that leverages Ollama's large language models to
provide intelligent, actionable feedback on GitHub pull requests.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Overview

This script automates the code review process by:
1. Fetching pull request changes from GitHub
2. Analyzing each file's diff using Ollama's AI models
3. Generating structured, actionable feedback
4. Posting formatted review comments directly to the PR

## Features

- **AI-Powered Reviews**: Uses Ollama's large language models for intelligent
  code analysis
- **Focused Feedback**: Identifies only serious or glaring issues, avoiding
  nitpicking
- **Structured Output**: Returns reviews in JSON format with overall comments
  and specific suggestions
- **Binary File Filtering**: Automatically skips non-text files
- **Long Comment Handling**: Automatically splits reviews that exceed GitHub's
  comment length limits
- **Formatted Output**: Reviews are formatted to 80 columns for readability
- **Django Test Awareness**: Ignores hardcoded test values in Django test cases
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Requirements

### Software Dependencies

- Python 3.8 or higher
- Ollama server (running and accessible)
- Git (for repository operations)

### Python Packages

```
PyGithub>=2.0.0
requests>=2.28.0
python-dotenv>=0.19.0
ollama>=0.1.0
```

Install via pip:
```bash
pip install PyGithub requests python-dotenv ollama
```

### Authentication

- **GitHub Token**: Personal access token with `repo` scope
- **Ollama Access**: Network access to Ollama server

## Installation

1. Clone or download this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables (see [Configuration](#configuration))

## Configuration

### Environment Variables

Create a `.env` file in the script directory:

```bash
# Required
GITHUB_TOKEN=ghp_your_github_token_here
GITHUB_REPO=owner/repository

# Optional (defaults shown)
OLLAMA_API_URL=http://192.168.3.16
OLLAMA_MODEL=gpt-oss:120b
```

### GitHub Token Setup

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token (classic)
3. Select scopes: `repo` (full control of private repositories)
4. Copy the token and add it to your `.env` file

### Ollama Setup

1. Install Ollama from https://ollama.ai
2. Start the Ollama service
3. Pull your desired model:
   ```bash
   ollama pull gpt-oss:120b
   # or
   ollama pull codellama:13b
   ```
4. Verify the service is running:
   ```bash
   curl http://localhost:11434/api/tags
   ```

## Usage

### Basic Usage

Review a pull request:
```bash
python github_pr_review_ollama.py PR_NUMBER
```

Example:
```bash
python github_pr_review_ollama.py 123
```

### Advanced Usage

Specify a different repository:
```bash
python github_pr_review_ollama.py 123 --repo owner/repository
```

Use a different Ollama model:
```bash
python github_pr_review_ollama.py 123 --model codellama:13b
```

Specify a different Ollama host:
```bash
python github_pr_review_ollama.py 123 --ollama_host http://localhost:11434
```

Combine options:
```bash
python github_pr_review_ollama.py 123 \
  --repo owner/repository \
  --model codellama:13b \
  --ollama_host http://localhost:11434
```

### Command-Line Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `pr_number` | Yes | Pull request number to review | - |
| `--repo` | No | Repository in `owner/repo` format | From `GITHUB_REPO` env var |
| `--model` | No | Ollama model name | From `OLLAMA_MODEL` env var or `gpt-oss:120b` |
| `--ollama_host` | No | Ollama server URL | From `OLLAMA_API_URL` env var or `http://192.168.3.16` |

## How It Works

### Workflow

1. **Initialization**
   - Loads environment variables
   - Validates Ollama connection and model availability
   - Authenticates with GitHub

2. **PR Retrieval**
   - Fetches the specified pull request
   - Retrieves all changed files and their diffs

3. **File Processing**
   - Filters out binary files (images, PDFs, etc.)
   - Skips files with no diff content

4. **AI Review**
   - Sends each file's diff to Ollama with a structured prompt
   - Receives JSON-formatted review with suggestions and comments

5. **Comment Formatting**
   - Parses JSON responses
   - Formats text to 80 columns for readability
   - Combines all file reviews into a single comment

6. **Posting**
   - Posts review to GitHub as a PR comment
   - Splits into multiple comments if exceeding length limits

### Review Prompt

The script instructs the AI to:
- Focus on serious issues only (not nitpicking)
- Provide specific, actionable suggestions
- Include line numbers where applicable
- Format responses as structured JSON
- Ignore hardcoded test values in Django tests

### Output Format

Reviews are structured as:

```markdown
## Code Review for PR #123

Hey @author, here's my review of your pull request:

### filename.py

**Overall Comments:**

General observations about the file changes...

**Suggestions:**

1. Specific suggestion with proper formatting
   and line wrapping for readability.

2. Another suggestion with details about what
   should be improved.

---
This review was generated by an automated code review bot using gpt-oss:120b.
```

## Customization

### Constants

You can modify these constants in [github_pr_review_ollama.py](../github_pr_review_ollama.py):

```python
DEFAULT_OLLAMA_HOST = 'http://192.168.3.16'      # Default Ollama server
DEFAULT_OLLAMA_MODEL = 'gpt-oss:120b'            # Default model
MAX_COMMENT_LENGTH = 65000                        # GitHub comment limit
TEXT_WIDTH = 80                                   # Output formatting width
OLLAMA_TIMEOUT = 5                                # Connection timeout (seconds)
OLLAMA_TEMPERATURE = 0.2                          # Model creativity (0-1)
OLLAMA_CONTEXT_SIZE = 8192                        # Context window size
BINARY_FILE_EXTENSIONS = ('.png', '.jpg', ...)   # Files to skip
```

### Review Prompt Customization

To modify the review behavior, edit the `prompt` in the `get_ollama_review()`
function at [github_pr_review_ollama.py:67-82](../github_pr_review_ollama.py#L67-L82):

```python
prompt = (
    "You are an expert code reviewer..."
    # Modify instructions here
)
```

### Adding Custom File Filters

To skip additional file types, update the `is_binary_file()` function at
[github_pr_review_ollama.py:245-254](../github_pr_review_ollama.py#L245-L254):

```python
def is_binary_file(filename: str) -> bool:
    excluded_extensions = ('.png', '.jpg', '.md', '.txt')  # Add more
    excluded_patterns = ('test_', 'mock_')  # Add patterns

    return (filename.endswith(excluded_extensions) or
            any(pattern in filename for pattern in excluded_patterns))
```

## Troubleshooting

### Connection Issues

**Problem**: `Could not connect to Ollama at <host>`

**Solutions**:
1. Verify Ollama is running:
   ```bash
   curl http://localhost:11434/api/tags
   ```
2. Check firewall settings
3. Verify the `OLLAMA_API_URL` is correct
4. Try running the script on the same machine as Ollama

### Model Not Available

**Problem**: `Model <name> is not available`

**Solutions**:
1. List available models:
   ```bash
   ollama list
   ```
2. Pull the required model:
   ```bash
   ollama pull gpt-oss:120b
   ```

### GitHub Authentication Errors

**Problem**: `Error accessing repository: 401 Unauthorized`

**Solutions**:
1. Verify your `GITHUB_TOKEN` is set correctly
2. Check the token has `repo` scope
3. Ensure the token hasn't expired
4. Verify you have access to the repository

### Review Comment Not Posted

**Problem**: Review generated but not posted to GitHub

**Solutions**:
1. Check GitHub API rate limits
2. Verify the PR is still open
3. Check log output for specific errors
4. Ensure the token has write permissions

### JSON Parsing Errors

**Problem**: `Failed to parse JSON response`

**Solutions**:
1. This is usually due to the model not following the JSON format
2. The script will fall back to posting the raw response
3. Try adjusting `OLLAMA_TEMPERATURE` to 0.1 for more consistent output
4. Try a different model that's better at structured output

## API Reference

### Functions

#### `check_ollama_connection(ollama_host: str) -> bool`
Verifies Ollama server is accessible.

**Parameters:**
- `ollama_host`: URL of the Ollama server

**Returns:** `True` if connection successful, `False` otherwise

**Location:** [github_pr_review_ollama.py:46-60](../github_pr_review_ollama.py#L46-L60)

---

#### `get_ollama_review(diff_text: str, filename: str, model: str) -> str`
Requests code review from Ollama for a specific file.

**Parameters:**
- `diff_text`: Git diff content for the file
- `filename`: Name of the file being reviewed
- `model`: Ollama model name to use

**Returns:** JSON string with review or error message

**Location:** [github_pr_review_ollama.py:62-105](../github_pr_review_ollama.py#L62-L105)

**Example:**
```python
diff = "@@ -10,3 +10,5 @@ def hello():\n+    print('world')"
review = get_ollama_review(diff, "app.py", "gpt-oss:120b")
```

---

#### `format_review_response(json_response: Dict[str, Any]) -> str`
Formats JSON review into markdown.

**Parameters:**
- `json_response`: Dictionary with `overall_comments` and `suggestions` keys

**Returns:** Formatted markdown string

**Location:** [github_pr_review_ollama.py:107-143](../github_pr_review_ollama.py#L107-L143)

**Example:**
```python
response = {
    "overall_comments": "Good changes overall.",
    "suggestions": ["Add error handling", "Use type hints"]
}
formatted = format_review_response(response)
```

---

#### `validate_environment(ollama_host: str, ollama_model: str) -> None`
Validates Ollama connection and model availability.

**Parameters:**
- `ollama_host`: Ollama server URL
- `ollama_model`: Model name to validate

**Raises:** `SystemExit` if validation fails

**Location:** [github_pr_review_ollama.py:146-179](../github_pr_review_ollama.py#L146-L179)

---

#### `setup_github_client(github_token: str, repo_name: str) -> Repository`
Initializes GitHub client and retrieves repository.

**Parameters:**
- `github_token`: GitHub authentication token
- `repo_name`: Repository in `owner/repo` format

**Returns:** GitHub Repository object

**Raises:** `SystemExit` if setup fails

**Location:** [github_pr_review_ollama.py:182-203](../github_pr_review_ollama.py#L182-L203)

---

#### `is_binary_file(filename: str) -> bool`
Checks if file is binary based on extension.

**Parameters:**
- `filename`: Name of the file

**Returns:** `True` if binary, `False` otherwise

**Location:** [github_pr_review_ollama.py:206-216](../github_pr_review_ollama.py#L206-L216)

**Example:**
```python
is_binary_file("image.png")  # Returns True
is_binary_file("script.py")  # Returns False
```

---

#### `process_review_response(review_response: str, filename: str) -> str`
Processes and formats Ollama's review response.

**Parameters:**
- `review_response`: Raw response from Ollama
- `filename`: Name of the file reviewed

**Returns:** Formatted review string

**Location:** [github_pr_review_ollama.py:219-238](../github_pr_review_ollama.py#L219-L238)

---

#### `post_review_comment(pr: PullRequest, combined_review: str, pr_author: str, pr_number: int) -> None`
Posts review comment to GitHub PR.

**Parameters:**
- `pr`: GitHub PullRequest object
- `combined_review`: Formatted review text
- `pr_author`: Username of PR author
- `pr_number`: PR number

**Location:** [github_pr_review_ollama.py:241-272](../github_pr_review_ollama.py#L241-L272)

**Behavior:**
- Automatically splits comments exceeding `MAX_COMMENT_LENGTH`
- Posts multiple comments if needed
- Removes duplicate mentions in continuation comments

---

### Constants Reference

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `DEFAULT_OLLAMA_HOST` | str | `'http://192.168.3.16'` | Default Ollama server URL |
| `DEFAULT_OLLAMA_MODEL` | str | `'gpt-oss:120b'` | Default model name |
| `MAX_COMMENT_LENGTH` | int | `65000` | Max GitHub comment length |
| `TEXT_WIDTH` | int | `80` | Text wrapping width |
| `OLLAMA_TIMEOUT` | int | `5` | Connection timeout (seconds) |
| `OLLAMA_TEMPERATURE` | float | `0.2` | Model temperature setting |
| `OLLAMA_CONTEXT_SIZE` | int | `8192` | Model context window size |
| `BINARY_FILE_EXTENSIONS` | tuple | `('.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip')` | File extensions to skip |

**Location:** [github_pr_review_ollama.py:35-42](../github_pr_review_ollama.py#L35-L42)

---

## Examples

### Example 1: Review a Single PR

```bash
# Set up environment
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxx
export GITHUB_REPO=owner/repository

# Review PR #123
python github_pr_review_ollama.py 123
```

### Example 2: Use Different Model

```bash
# Use CodeLlama for review
python github_pr_review_ollama.py 456 --model codellama:13b
```

### Example 3: Custom Ollama Server

```bash
# Use Ollama running on a different machine
python github_pr_review_ollama.py 789 \
  --ollama_host http://10.0.1.5:11434
```

### Example 4: Complete Workflow

```bash
# Complete setup and review
cat > .env << EOF
GITHUB_TOKEN=ghp_your_token_here
GITHUB_REPO=myorg/myrepo
OLLAMA_API_URL=http://localhost:11434
OLLAMA_MODEL=codellama:13b
EOF

# Install dependencies
pip install PyGithub requests python-dotenv ollama

# Start Ollama and pull model
ollama serve &
ollama pull codellama:13b

# Run review
python github_pr_review_ollama.py 100
```

### Example 5: Integration with CI/CD

Create a GitHub Actions workflow:

```yaml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install PyGithub requests python-dotenv ollama

      - name: Run AI Review
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_REPO: ${{ github.repository }}
          OLLAMA_API_URL: ${{ secrets.OLLAMA_API_URL }}
          OLLAMA_MODEL: codellama:13b
        run: |
          python github_pr_review_ollama.py ${{ github.event.pull_request.number }}
```

---

## Best Practices

### Model Selection

- **Small PRs (<500 lines)**: Use `codellama:7b` for faster reviews
- **Medium PRs (500-2000 lines)**: Use `codellama:13b` for balanced performance
- **Large PRs (>2000 lines)**: Use `gpt-oss:120b` or similar for comprehensive analysis
- **Specialized code**: Use domain-specific models (e.g., `starcoder` for general code)

### Performance Optimization

1. **Run Ollama locally** when possible for faster response times
2. **Use SSD storage** for Ollama model cache
3. **Allocate sufficient RAM** (at least 16GB for larger models)
4. **Adjust context size** based on your typical file sizes

### Review Quality

1. **Keep PRs small**: Reviews are most effective for PRs under 1000 lines
2. **Review incrementally**: Run reviews on each commit for ongoing feedback
3. **Combine with human review**: AI reviews complement, not replace, human reviewers
4. **Tune the prompt**: Adjust the review prompt for your team's coding standards

### Security Considerations

1. **Protect your GitHub token**: Never commit tokens to version control
2. **Use environment variables**: Store sensitive data in `.env` files
3. **Restrict token scope**: Use tokens with minimal required permissions
4. **Review generated content**: Always verify AI suggestions before applying

---

## Advanced Usage

### Custom Review Prompts

Create specialized review prompts for different file types:

```python
def get_custom_prompt(filename: str) -> str:
    if filename.endswith('.py'):
        return "Focus on Python best practices, PEP 8, and type hints..."
    elif filename.endswith('.js'):
        return "Focus on JavaScript best practices, ESLint rules..."
    elif filename.endswith('.sql'):
        return "Focus on SQL injection risks, query optimization..."
    else:
        return "Provide general code review feedback..."
```

### Filtering Reviews by Severity

Modify the prompt to categorize issues:

```python
prompt = (
    "Categorize each issue as CRITICAL, HIGH, MEDIUM, or LOW. "
    "Return JSON with structure: "
    "{'critical': [...], 'high': [...], 'medium': [...], 'low': [...]}"
)
```

### Integration with Other Tools

Combine with other code analysis tools:

```bash
# Run linter first
pylint myfile.py > lint_report.txt

# Then run AI review
python github_pr_review_ollama.py 123

# Combine results
cat lint_report.txt >> review_combined.txt
```

---

## Limitations

1. **Model Context Window**: Large files may exceed context limits
2. **Processing Time**: Reviews can take several minutes for large PRs
3. **JSON Formatting**: Some models may not always return valid JSON
4. **False Positives**: AI may flag valid patterns as issues
5. **Language Support**: Review quality varies by programming language
6. **Network Dependency**: Requires stable connection to Ollama server

---

## FAQ

**Q: Can I use this with private repositories?**
A: Yes, ensure your GitHub token has access to private repositories.

**Q: Does this work with GitHub Enterprise?**
A: Yes, you may need to configure the GitHub API endpoint.

**Q: Can I review multiple PRs at once?**
A: Not directly, but you can write a wrapper script to iterate over PR numbers.

**Q: How much does this cost?**
A: The script is free. You only need compute resources for running Ollama.

**Q: Can I run this in CI/CD?**
A: Yes, see the CI/CD example above. Ensure Ollama is accessible from your CI environment.

**Q: What if the review is too verbose?**
A: Adjust the prompt to request more concise feedback or increase `OLLAMA_TEMPERATURE`.

**Q: Can I review files in specific directories only?**
A: Yes, add filtering logic in the file processing loop:
```python
if not file.filename.startswith('src/'):
    continue
```

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with clear commit messages
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/repo.git
cd repo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

---

## License

[Specify your license here - e.g., MIT, Apache 2.0, GPL]

---

## Changelog

### Version 1.0.0 (Current)
- Initial release
- Support for Ollama-based code reviews
- Automatic comment splitting for long reviews
- Binary file filtering
- Django test case handling
- Comprehensive error handling and logging

---

## Support

For issues and questions:

- **Bug Reports**: Open an issue on GitHub
- **Feature Requests**: Open an issue with the "enhancement" label
- **Questions**: Check existing issues or start a discussion
- **Security Issues**: Report privately to maintainers

---

## Acknowledgments

- [Ollama](https://ollama.ai) for providing the LLM infrastructure
- [PyGithub](https://github.com/PyGithub/PyGithub) for the GitHub API wrapper
- The open-source community for inspiration and best practices

---

**Last Updated**: 2026-01-06
**Maintainer**: [Your Name/Organization]
**Version**: 1.0.0
