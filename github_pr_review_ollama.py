#!/usr/bin/env python3

# GitHub PR Code Reviewer using Ollama
# This script reviews GitHub pull requests using Ollama's large language models
# to provide code review feedback directly as PR comments.
#
# Usage:
#   python pr_reviewer.py PR_NUMBER [--repo OWNER/REPO] [--model MODEL_NAME]
#
# Requirements:
#   - GitHub token with repo access (set as GITHUB_TOKEN environment variable)
#   - Ollama running and accessible (set OLLAMA_API_URL if not default)
#   - Python packages: PyGithub, requests, python-dotenv, ollama

import os
import sys
import argparse
import json
import logging
import textwrap
from typing import Dict, List, Optional, Any
import requests
from dotenv import load_dotenv
from github import Github, Auth, PullRequest, Repository
import ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OLLAMA_HOST = 'http://192.168.3.16'
DEFAULT_OLLAMA_MODEL = 'gpt-oss:120b'
MAX_COMMENT_LENGTH = 65000
TEXT_WIDTH = 80
OLLAMA_TIMEOUT = 5
OLLAMA_TEMPERATURE = 0.2
OLLAMA_CONTEXT_SIZE = 8192
BINARY_FILE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip')

def check_ollama_connection(ollama_host: str) -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=OLLAMA_TIMEOUT)
        if response.status_code == 200:
            logger.info(f"Successfully connected to Ollama at {ollama_host}")
            return True
        # If that fails, try the Python SDK
        os.environ['OLLAMA_HOST'] = ollama_host
        ollama.list()
        logger.info(f"Successfully connected to Ollama at {ollama_host} via SDK")
        return True
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        return False

def get_ollama_review(diff_text: str, filename: str, model: str) -> str:
    """Get code review from Ollama for a specific file diff.

    Args:
        diff_text: The git diff text for the file
        filename: Name of the file being reviewed
        model: Ollama model name to use for review

    Returns:
        JSON string containing review suggestions and comments, or error message
    """
    prompt = (
        "You are an expert code reviewer with deep knowledge of software engineering best practices. "
        f"Review the following code changes for the file: {filename}. "
        "Provide specific, actionable suggestions for improvement, including line numbers where applicable. "
        "Focus on glaring or serious issues only. "
        "If there are no significant issues, state 'No significant issues found in this file.' "
        "Format your response as a JSON object with keys: "
        f"'suggestions' (a list of strings, each suggestion should be properly formatted to {TEXT_WIDTH} columns or less "
        f"with newlines for readability) and 'overall_comments' (a string with proper formatting to {TEXT_WIDTH} columns "
        "or less, using newlines for paragraph breaks). "
        f"Each suggestion should be a complete thought that fits within {TEXT_WIDTH} columns. "
        "Do not include any other text outside this JSON object."
        "\n\nNote: Ignore hardcoded values used for testing in Django test cases, as they are needed to drive tests."
    )
    try:
        logger.info(f"Requesting review for {filename} using model {model}")
        response = ollama.generate(
            model=model,
            prompt=f"{prompt}\n\nCode changes:\n{diff_text}",
            options={
                'temperature': OLLAMA_TEMPERATURE,
                'num_ctx': OLLAMA_CONTEXT_SIZE
            }
        )
        return response['response']
    except Exception as e:
        error_msg = f"Error reviewing {filename}: {str(e)}"
        logger.error(error_msg)
        return error_msg

def format_review_response(json_response: Dict[str, Any]) -> str:
    """Format the JSON response into a nicely formatted markdown string.

    Args:
        json_response: Dictionary containing 'overall_comments' and 'suggestions' keys

    Returns:
        Formatted markdown string with review comments
    """
    if not json_response:
        return "No review comments provided."

    # Format overall comments with proper line wrapping
    overall_comments = json_response.get('overall_comments', '')
    if overall_comments:
        paragraphs = overall_comments.split('\n\n')
        wrapped_paragraphs = [
            textwrap.fill(para, width=TEXT_WIDTH, initial_indent='', subsequent_indent='  ')
            for para in paragraphs if para.strip()
        ]
        formatted_comments = '\n\n'.join(wrapped_paragraphs)
    else:
        formatted_comments = "No overall comments provided."

    # Format suggestions
    suggestions = json_response.get('suggestions', [])
    if suggestions:
        formatted_suggestions = []
        for i, suggestion in enumerate(suggestions, 1):
            paragraphs = suggestion.split('\n\n')
            wrapped_paragraphs = [
                textwrap.fill(para, width=TEXT_WIDTH, initial_indent='  ', subsequent_indent='    ')
                for para in paragraphs if para.strip()
            ]
            formatted_suggestion = '\n\n'.join(wrapped_paragraphs)
            formatted_suggestions.append(f"{i}. {formatted_suggestion}")
        formatted_suggestions_str = "\n\n".join(formatted_suggestions)
        return f"**Overall Comments:**\n\n{formatted_comments}\n\n**Suggestions:**\n\n{formatted_suggestions_str}"
    else:
        return f"**Overall Comments:**\n\n{formatted_comments}"


def validate_environment(ollama_host: str, ollama_model: str) -> None:
    """Validate Ollama connection and model availability.

    Args:
        ollama_host: Ollama server URL
        ollama_model: Model name to validate

    Raises:
        SystemExit: If validation fails
    """
    if not check_ollama_connection(ollama_host):
        logger.error(f"Could not connect to Ollama at {ollama_host}")
        print(f"\nError: Could not connect to Ollama at {ollama_host}")
        print("\nDebugging information:")
        print("1. Confirm Ollama is running")
        print("2. Possible issues:")
        print("   - The API endpoint might be different than expected")
        print("   - The Python SDK might need specific configuration")
        print("   - There might be network-level restrictions")
        print("\nTry these solutions:")
        print("1. Verify the API endpoint works:")
        print(f"   curl {ollama_host}/api/tags")
        print("2. Check if the model is available:")
        print(f"   curl {ollama_host}/api/show -d '{{\"name\": \"{ollama_model}\"}}'")
        print("3. Try running the script on the same machine as Ollama")
        sys.exit(1)

    try:
        os.environ['OLLAMA_HOST'] = ollama_host
        models = ollama.list()['models']
        available_models = [m['model'] for m in models]
        if ollama_model not in available_models:
            logger.error(f"Model {ollama_model} is not available")
            print(f"\nError: Model {ollama_model} is not available")
            print(f"Available models: {', '.join(available_models)}")
            print(f"To download the model, run: ollama pull {ollama_model}")
            sys.exit(1)
        logger.info(f"Model {ollama_model} is available")
    except Exception as e:
        logger.error(f"Error checking available models: {str(e)}")
        print(f"\nError checking available models: {str(e)}")
        sys.exit(1)


def setup_github_client(github_token: str, repo_name: str) -> tuple[Repository.Repository, PullRequest.PullRequest, int]:
    """Initialize GitHub client and retrieve PR.

    Args:
        github_token: GitHub authentication token
        repo_name: Repository name in format 'owner/repo'

    Returns:
        Tuple of (repository, pull_request, pr_number)

    Raises:
        SystemExit: If setup fails
    """
    g = Github(auth=Auth.Token(github_token))

    try:
        repo = g.get_repo(repo_name)
        return repo
    except Exception as e:
        logger.error(f"Error accessing repository: {str(e)}")
        print(f"Error accessing repository or PR: {str(e)}")
        sys.exit(1)


def is_binary_file(filename: str) -> bool:
    """Check if a file is a binary file based on its extension.

    Args:
        filename: Name of the file to check

    Returns:
        True if the file is binary, False otherwise
    """
    return filename.endswith(BINARY_FILE_EXTENSIONS)


def process_review_response(review_response: str, filename: str) -> str:
    """Process Ollama review response and format it.

    Args:
        review_response: Raw response from Ollama
        filename: Name of the file being reviewed

    Returns:
        Formatted review string
    """
    if review_response.startswith("Error"):
        return f"### {filename}\n\n{review_response}\n"

    try:
        json_response = json.loads(review_response)
        formatted_review = format_review_response(json_response)
        return f"### {filename}\n\n{formatted_review}\n"
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON response for {filename}, using raw response")
        return f"### {filename}\n\n{review_response}\n"


def post_review_comment(pr: PullRequest.PullRequest, combined_review: str, pr_author: str, pr_number: int) -> None:
    """Post review comment to GitHub PR, splitting if necessary.

    Args:
        pr: GitHub PullRequest object
        combined_review: The formatted review text
        pr_author: Username of the PR author
        pr_number: PR number for logging
    """
    if len(combined_review) > MAX_COMMENT_LENGTH:
        logger.info(f"Comment exceeds {MAX_COMMENT_LENGTH} characters, splitting into chunks")
        print("Comment too long. Splitting into multiple comments...")
        chunks = [combined_review[i:i+MAX_COMMENT_LENGTH] for i in range(0, len(combined_review), MAX_COMMENT_LENGTH)]

        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"Posting review part {i+1}/{len(chunks)}")
                print(f"Review Part {i+1}:\n\n{chunk}")
                if i == 0:
                    pr.create_issue_comment(chunk)
                else:
                    chunk_without_mention = chunk.replace(f"Hey @{pr_author}, here's my review of your pull request:\n\n", "")
                    pr.create_issue_comment(f"Review Part {i+1} (continued):\n\n{chunk_without_mention}")
            except Exception as e:
                logger.error(f"Error posting comment part {i+1}: {str(e)}")
                print(f"Error posting comment part {i+1}: {str(e)}")
    else:
        try:
            print(combined_review)
            pr.create_issue_comment(combined_review)
            logger.info(f"Successfully posted review for PR #{pr_number}")
        except Exception as e:
            logger.error(f"Error posting comment: {str(e)}")
            print(f"Error posting comment: {str(e)}")
            sys.exit(1)


def main() -> None:
    """Main function to orchestrate the PR review process."""
    load_dotenv()
    logger.info("Starting GitHub PR review script")

    # Configure Ollama settings
    ollama_host = os.getenv('OLLAMA_API_URL', DEFAULT_OLLAMA_HOST)
    ollama_model = os.getenv('OLLAMA_MODEL', DEFAULT_OLLAMA_MODEL)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Review GitHub PR using Ollama')
    parser.add_argument('pr_number', type=int, help='Pull request number to review')
    parser.add_argument('--repo', help='GitHub repository (owner/repo)', default=os.getenv('GITHUB_REPO'))
    parser.add_argument('--model', help='Ollama model to use', default=ollama_model)
    parser.add_argument('--ollama_host', help='Ollama host URL', default=ollama_host)
    args = parser.parse_args()

    # Override with command-line arguments if provided
    ollama_host = args.ollama_host
    ollama_model = args.model
    os.environ['OLLAMA_HOST'] = ollama_host

    # Validate environment and credentials
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        logger.error("GITHUB_TOKEN not found")
        print("Error: GITHUB_TOKEN must be set in .env or as environment variable")
        sys.exit(1)
    if not args.repo:
        logger.error("GITHUB_REPO not provided")
        print("Error: GITHUB_REPO must be set in .env or provided via --repo argument")
        sys.exit(1)

    # Validate Ollama environment
    validate_environment(ollama_host, ollama_model)

    # Setup GitHub client and get PR
    repo = setup_github_client(github_token, args.repo)

    try:
        pr = repo.get_pull(args.pr_number)
        pr_author = pr.user.login
        logger.info(f"Retrieved PR #{args.pr_number} by @{pr_author}")
    except Exception as e:
        logger.error(f"Error retrieving PR #{args.pr_number}: {str(e)}")
        print(f"Error accessing PR: {str(e)}")
        sys.exit(1)

    # Get and validate changed files
    try:
        files = pr.get_files()
    except Exception as e:
        logger.error(f"Error getting PR files: {str(e)}")
        print(f"Error getting PR files: {str(e)}")
        sys.exit(1)

    if not files:
        logger.info("No files changed in this PR")
        print("No files changed in this PR.")
        return

    # Process each file
    reviews: List[str] = []
    for file in files:
        if not file.patch:
            logger.debug(f"Skipping {file.filename} (no diff)")
            continue

        if is_binary_file(file.filename):
            logger.info(f"Skipping binary file: {file.filename}")
            print(f"Skipping binary file: {file.filename}")
            continue

        logger.info(f"Reviewing {file.filename}")
        print(f"Reviewing {file.filename}...")
        review_response = get_ollama_review(file.patch, file.filename, ollama_model)
        formatted = process_review_response(review_response, file.filename)
        reviews.append(formatted)

    if not reviews:
        logger.info("No reviewable changes found")
        print("No reviewable changes found in this PR.")
        return

    # Combine all reviews
    combined_review = (
        f"## Code Review for PR #{args.pr_number}\n\n"
        f"Hey @{pr_author}, here's my review of your pull request:\n\n"
        + "\n".join(reviews) +
        "\n\n---\n"
        f"This review was generated by an automated code review bot using {ollama_model}."
    )

    # Post the review
    post_review_comment(pr, combined_review, pr_author, args.pr_number)
    print("Review submitted successfully!")

if __name__ == "__main__":
    main()

