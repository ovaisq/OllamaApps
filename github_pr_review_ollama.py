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
import textwrap
import requests
from dotenv import load_dotenv
from github import Github
import ollama

def check_ollama_connection(ollama_host):
    """Check if Ollama is running and accessible on port 80"""
    try:
        # First try direct HTTP request to port 80
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            return True
        # If that fails, try the Python SDK
        os.environ['OLLAMA_HOST'] = ollama_host
        ollama.list()
        return True
    except Exception as e:
        print(f"Connection test failed: {str(e)}")
        return False

def get_ollama_review(diff_text, filename, model):
    prompt = (
        "You are an expert code reviewer with deep knowledge of software engineering best practices. "
        f"Review the following code changes for the file: {filename}. "
        "Provide specific, actionable suggestions for improvement, including line numbers where applicable. "
        "Focus on glaring or serious issues only. "
        "If there are no significant issues, state 'No significant issues found in this file.' "
        "Format your response as a JSON object with keys: "
        "'suggestions' (a list of strings, each suggestion should be properly formatted to 80 columns or less "
        "with newlines for readability) and 'overall_comments' (a string with proper formatting to 80 columns "
        "or less, using newlines for paragraph breaks). "
        "Each suggestion should be a complete thought that fits within 80 columns. "
        "Do not include any other text outside this JSON object."
    )
    try:
        response = ollama.generate(
            model=model,
            prompt=f"{prompt}\n\nCode changes:\n{diff_text}",
            options={
                'temperature': 0.2,
                'num_ctx': 8192
            }
        )
        return response['response']
    except Exception as e:
        return f"Error reviewing {filename}: {str(e)}"

def format_review_response(json_response):
    """Format the JSON response into a nicely formatted markdown string"""
    if not json_response:
        return "No review comments provided."

    # Format overall comments with proper line wrapping
    overall_comments = json_response.get('overall_comments', '')
    if overall_comments:
        # Split into paragraphs and wrap each
        paragraphs = overall_comments.split('\n\n')
        wrapped_paragraphs = []
        for para in paragraphs:
            if para.strip():
                wrapped = textwrap.fill(para, width=80, initial_indent='', subsequent_indent='  ')
                wrapped_paragraphs.append(wrapped)
        formatted_comments = '\n\n'.join(wrapped_paragraphs)
    else:
        formatted_comments = "No overall comments provided."

    # Format suggestions
    suggestions = json_response.get('suggestions', [])
    if suggestions:
        formatted_suggestions = []
        for i, suggestion in enumerate(suggestions, 1):
            # Split into paragraphs and wrap each
            paragraphs = suggestion.split('\n\n')
            wrapped_paragraphs = []
            for para in paragraphs:
                if para.strip():
                    wrapped = textwrap.fill(para, width=80, initial_indent='  ',
                                          subsequent_indent='    ')
                    wrapped_paragraphs.append(wrapped)
            formatted_suggestion = '\n\n'.join(wrapped_paragraphs)
            formatted_suggestions.append(f"{i}. {formatted_suggestion}")
        formatted_suggestions_str = "\n\n".join(formatted_suggestions)
        return f"**Overall Comments:**\n\n{formatted_comments}\n\n**Suggestions:**\n\n{formatted_suggestions_str}"
    else:
        return f"**Overall Comments:**\n\n{formatted_comments}"

def main():
    # Load environment variables
    load_dotenv()

    # Configure Ollama client host - explicitly using port 80
    ollama_host = os.getenv('OLLAMA_API_URL', 'http://192.168.3.16')
    ollama_model = os.getenv('OLLAMA_MODEL', 'gpt-oss:120b')

    # Force the OLLAMA_HOST environment variable
    os.environ['OLLAMA_HOST'] = ollama_host

    # Check Ollama connection first
    if not check_ollama_connection(ollama_host):
        print(f"\nError: Could not connect to Ollama at {ollama_host}")
        print("\nDebugging information:")
        print("1. Confirmed Ollama is running on port 80 (via curl)")
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

    # Verify the model is available
    try:
        os.environ['OLLAMA_HOST'] = ollama_host
        models = ollama.list()['models']
        available_models = [m['model'] for m in models]
        if ollama_model not in available_models:
            print(f"\nError: Model {ollama_model} is not available")
            print(f"Available models: {', '.join(available_models)}")
            print(f"To download the model, run: ollama pull {ollama_model}")
            sys.exit(1)
    except Exception as e:
        print(f"\nError checking available models: {str(e)}")
        sys.exit(1)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Review GitHub PR using Ollama with gpt-oss:120b')
    parser.add_argument('pr_number', type=int, help='Pull request number to review')
    parser.add_argument('--repo', help='GitHub repository (owner/repo)', default=os.getenv('GITHUB_REPO'))
    parser.add_argument('--model', help='Ollama model to use', default=ollama_model)
    args = parser.parse_args()

    # Validate required arguments
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("Error: GITHUB_TOKEN must be set in .env or as environment variable")
        sys.exit(1)
    if not args.repo:
        print("Error: GITHUB_REPO must be set in .env or provided via --repo argument")
        sys.exit(1)

    # Initialize GitHub client
    g = Github(github_token)
    try:
        repo = g.get_repo(args.repo)
        pr = repo.get_pull(args.pr_number)
    except Exception as e:
        print(f"Error accessing repository or PR: {str(e)}")
        sys.exit(1)

    # Get the PR author's username
    pr_author = pr.user.login

    # Get changed files
    try:
        files = pr.get_files()
    except Exception as e:
        print(f"Error getting PR files: {str(e)}")
        sys.exit(1)

    if not files:
        print("No files changed in this PR.")
        return

    # Process each file
    reviews = []
    for file in files:
        diff_text = file.patch
        if not diff_text:
            continue
        # Skip binary files
        if file.status == 'added' and file.filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip')):
            print(f"Skipping binary file: {file.filename}")
            continue
        print(f"Reviewing {file.filename}...")
        review_response = get_ollama_review(diff_text, file.filename, args.model)
        # Process the response
        if not review_response.startswith("Error"):
            try:
                json_response = json.loads(review_response)
                formatted_review = format_review_response(json_response)
                reviews.append(f"### {file.filename}\n\n{formatted_review}\n")
            except json.JSONDecodeError:
                # If it's not valid JSON, just add the raw response
                reviews.append(f"### {file.filename}\n\n{review_response}\n")
        else:
            reviews.append(f"### {file.filename}\n\n{review_response}\n")

    if not reviews:
        print("No reviewable changes found in this PR.")
        return

    # Format combined review with author mention
    combined_review = (
        f"## Code Review for PR #{args.pr_number} (using {args.model})\n\n"
        f"Hey @{pr_author}, here's my review of your pull request:\n\n"
        + "\n".join(reviews) +
        "\n\n---\n"
        "This review was generated by an automated code review bot using Ollama."
    )

    # Handle GitHub comment length limit
    max_length = 65000
    if len(combined_review) > max_length:
        print("Comment too long. Splitting into multiple comments...")
        chunks = [combined_review[i:i+max_length] for i in range(0, len(combined_review), max_length)]
        for i, chunk in enumerate(chunks):
            try:
                print(f"Review Part {i+1}:\n\n{chunk}")
                if i == 0:  # Only mention the author in the first part
                    pr.create_issue_comment(chunk)
                else:
                    # For subsequent parts, remove the author mention to avoid duplicate notifications
                    chunk_without_mention = chunk.replace(f"Hey @{pr_author}, here's my review of your pull request:\n\n", "")
                    pr.create_issue_comment(f"Review Part {i+1} (continued):\n\n{chunk_without_mention}")
            except Exception as e:
                print(f"Error posting comment part {i+1}: {str(e)}")
    else:
        try:
            print(combined_review)
            pr.create_issue_comment(combined_review)
        except Exception as e:
            print(f"Error posting comment: {str(e)}")
            sys.exit(1)

    print("Review submitted successfully!")

if __name__ == "__main__":
    main()
