#!/usr/bin/env python3
"""
GitHub PR Code Reviewer using Ollama 
"""

import os
import sys
import argparse
import json
import textwrap
from typing import Dict, List, Optional, Any
import requests
from dotenv import load_dotenv
from github import Github
import ollama


class OllamaClient:
    """Handles Ollama connection and model operations"""
    
    def __init__(self, host: str, model: str):
        self.host = host
        self.model = model
        os.environ['OLLAMA_HOST'] = host
    
    def check_connection(self) -> bool:
        """Verify Ollama is accessible"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Connection test failed: {str(e)}")
            return False
    
    def verify_model(self) -> bool:
        """Check if specified model is available"""
        try:
            models = ollama.list()['models']
            available_models = [m['model'] for m in models]
            if self.model not in available_models:
                print(f"Model {self.model} not available. Existing models: {', '.join(available_models)}")
                return False
            return True
        except Exception as e:
            print(f"Error checking models: {str(e)}")
            return False


class ReviewGenerator:
    """Handles code review generation and formatting"""
    
    def __init__(self, model: str):
        self.model = model
    
    def _build_prompt(self, diff_text: str, filename: str) -> str:
        """Construct review prompt"""
        prompt_template = textwrap.dedent("""
            You are an expert code reviewer. Review code changes for {filename}.
            Provide specific, actionable suggestions. Focus on serious issues only.
            Format as JSON with keys: 'suggestions' (list of strings) and 'overall_comments' (string).
            Each suggestion should fit within 80 columns.
            Code changes:
            {diff_text}
        """)
        return prompt_template.format(filename=filename, diff_text=diff_text)
    
    def generate_review(self, diff_text: str, filename: str) -> Dict[str, Any]:
        """Generate review using Ollama"""
        try:
            prompt = self._build_prompt(diff_text, filename)
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={'temperature': 0.2, 'num_ctx': 8192}
            )
            return json.loads(response['response'])
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON response for {filename}"}
        except Exception as e:
            return ({"error": f"Error reviewing {filename}: {str(e)}"})
    
    def format_review(self, review_data: Dict[str, Any], filename: str) -> str:
        """Format review as markdown"""
        if "error" in review_data:
            return f"### {filename}\n\n{review_data['error']}\n"
        
        formatted = [f"### {filename}\n"]
        
        # Format overall comments
        if overall := review_data.get('overall_comments'):
            wrapped = self._wrap_text(overall)
            formatted.append(f"**Overall Comments:**\n\n{wrapped}\n")
        
        # Format suggestions
        if suggestions := review_data.get('suggestions', []):
            formatted.append("**Suggestions:**\n")
            for i, suggestion in enumerate(suggestions, 1):
                wrapped_suggestion = self._wrap_text(suggestion, indent='  ')
                formatted.append(f"{i}. {wrapped_suggestion}\n")
        
        return "\n".join(formatted)
    
    def _wrap_text(self, text: str, width: int = 80, indent: str = '') -> str:
        """Wrap text to specified width"""
        paragraphs = text.split('\n\n')
        wrapped_paragraphs = []
        for para in paragraphs:
            if para.strip():
                wrapped = textwrap.fill(
                    para, 
                    width=width, 
                    initial_indent=indent,
                    subsequent_indent=indent
                )
                wrapped_paragraphs.append(wrapped)
        return '\n\n'.join(wrapped_paragraphs)


class GitHubPRClient:
    """Handles GitHub PR operations"""
    
    def __init__(self, token: str, repo: str):
        self.github = Github(token)
        self.repo = self.github.get_repo(repo)
    
    def get_pr_details(self, pr_number: int):
        """Retrieve PR information"""
        pr = self.repo.get_pull(pr_number)
        return {
            "pr": pr,
            "author": pr.user.login,
            "files": pr.get_files()
        }


class Configuration:
    """Manages configuration loading and validation"""
    
    DEFAULT_CONFIG = {
        'OLLAMA_HOST': 'http://192.168.3.16',
        'OLLAMA_MODEL': 'gpt-oss:120b',
        'MAX_COMMENT_LENGTH': 65000
    }
    
    @classmethod
    def load(cls) -> Dict[str, str]:
        """Load configuration from environment"""
        load_dotenv()
        config = cls.DEFAULT_CONFIG.copy()
        
        # Override with environment variables
        for key in config:
            if env_value := os.getenv(key):
                config[key] = env_value
        
        return config
    
    @staticmethod
    def validate_required(required_vars: List[str]) -> bool:
        """Validate required environment variables"""
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            print(f"Error: Missing required variables: {', '.join(missing)}")
            return False
        return True


def split_long_comment(comment: str, max_length: int, author: str) -> List[str]:
    """Split long comments into multiple parts"""
    if len(comment) <= max_length:
        return [comment]
    
    chunks = []
    current_chunk = ""
    
    # Split by file sections to maintain logical boundaries
    file_sections = comment.split('### ')
    for section in file_sections:
        if not section:
            continue
        if len(current_chunk) + len(section) + 10 > max_length:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = f"### {section}"
        else:
            current_chunk += f"### {section}" if not current_chunk else section
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def main():
    # Load and validate configuration
    config = Configuration.load()
    
    if not Configuration.validate_required(['GITHUB_TOKEN', 'GITHUB_REPO']):
        sys.exit(1)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Review GitHub PR using Ollama')
    parser.add_argument('pr_number', type=int, help='Pull request number')
    parser.add_argument('--repo', default=os.getenv('GITHUB_REPO'))
    parser.add_argument('--model', default=config['OLLAMA_MODEL'])
    args = parser.parse_args()
    
    # Initialize clients
    ollama_client = OllamaClient(config['OLLAMA_HOST'], args.model)
    review_gen = ReviewGenerator(args.model)
    github_client = GitHubPRClient(
        os.getenv('GITHUB_TOKEN'),
        args.repo
    )
    
    # Check Ollama connection
    if not ollama_client.check_connection():
        print(f"Could not connect to Ollama at {config['OLLAMA_HOST']}")
        sys.exit(1)
    
    if not ollama_client.verify_model():
        sys.exit(1)
    
    # Get PR details
    try:
        pr_details = github_client.get_pr_details(args.pr_number)
    except Exception as e:
        print(f"Error accessing PR: {str(e)}")
        sys.exit(1)
    
    # Process files
    reviews = []
    for file in pr_details['files']:
        if not file.patch or file.filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip')):
            continue
        
        print(f"Reviewing {file.filename}...")
        review_data = review_gen.generate_review(file.patch, file.filename)
        formatted_review = review_gen.format_review(review_data, file.filename)
        reviews.append(formatted_review)
    
    if not reviews:
        print("No reviewable changes found.")
        return
    
    # Combine reviews
    combined = (
        f"## Code Review for PR #{args.pr_number} (using {args.model})\n\n"
        f"Hey @{pr_details['author']}, here's my review:\n\n"
        + "\n".join(reviews) +
        "\n\n---\nAutomated review using Ollama."
    )
    
    # Post review
    try:
        chunks = split_long_comment(combined, config['MAX_COMMENT_LENGTH'], pr_details['author'])
        
        for i, chunk in enumerate(chunks, 1):
            if i == 1:
                pr_details['pr'].create_issue_comment(chunk)
            else:
                # Remove author mention from subsequent chunks
                chunk = chunk.replace(
                    f"Hey @{pr_details['author']}, here's my review:\n\n",
                    ""
                )
                pr_details['pr'].create_issue_comment(f"Review Part {i}:\n\n{chunk}")
    except Exception as e:
        print(f"Error posting comment: {str(e)}")
        sys.exit(1)
    
    print("Review submitted successfully!")


if __name__ == "__main__":
    main()
