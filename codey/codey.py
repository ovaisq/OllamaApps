#!/usr/bin/env python3

import os
import glob
import json
import time
import re
import ollama
import chromadb
import concurrent.futures
import argparse

"""Script for indexing a Python codebase, searching it using embeddings,
    and interacting with an LLM to answer questions about the code.

    Features:
    - Indexes Python files from a specified directory.
    - Searches indexed files for relevant snippets based on a query.
    - Uses decision logic to determine whether to search or read files.
    - Interacts with an LLM to provide answers to user queries.
"""

OLLAMA_REMOTE_HOST = "http://"
OLLAMA_MODEL = "phi4-mini"

client = chromadb.Client()
collection = client.create_collection("codebase")

def extract_json(raw_text):
    # Try to extract JSON from ```json ... ``` fences
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1)
    # Fallback: find {...}
    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    return match.group(1) if match else raw_text

def embed_text(text: str):
    try:
        #model="nomic-embed-text",
        response = ollama.embed(
            model="granite-embedding:278m",
            input=text,
        )
        return response.get("embedding", [])
    except Exception as e:
        print(f"Embedding call error: {e}")
        return []

def process_chunk(file_path, chunk_idx, chunk):
    embedding = embed_text(chunk)
    if embedding:
        collection.add(
            ids=[f"{file_path}_{chunk_idx}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"file": file_path}]
        )

EXCLUDED_EXTENSIONS = {'.png', '.pem', '.jpg', '.log', '.yml', '.pyc', '.p8', '.gif', '.woff', '.eot', '.tiff', '.ttf', '.webp', '.svg', '.jpeg', '.ico', '.mp3'}
EXCLUDED_FILENAMES = {'Dockerfile'}

def is_excluded(file_path):
    _, ext = os.path.splitext(file_path)
    file_name = os.path.basename(file_path)
    return ext.lower() in EXCLUDED_EXTENSIONS or file_name in EXCLUDED_FILENAMES

def index_codebase(path):
    files_indexed = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for file_path in glob.glob(f"{path}/**/*.*", recursive=True):
            if is_excluded(file_path):
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            chunks = [content[i:i+500] for i in range(0, len(content), 500)]
            for idx, chunk in enumerate(chunks):
                futures.append(executor.submit(process_chunk, file_path, idx, chunk))
            files_indexed.append(file_path)
        # Wait for all futures to complete
        concurrent.futures.wait(futures)
    print(f"Indexed {len(files_indexed)} files.")
    return files_indexed

def search_code(query, k=3):
    query_embedding = embed_text(query)
    if not query_embedding:
        return []
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    docs = results["documents"][0]
    meta = results["metadatas"][0]
    return [(meta[i]["file"], docs[i]) for i in range(len(docs))]

def read_file(path):
    if not os.path.isfile(path):
        print(f"Skipping invalid path: {path}")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None

def list_files(path):
    return [f for f in glob.glob(f"{path}/**/*.py", recursive=True)]

def ollama_chat(model, messages):
    return ollama.chat(model=model, messages=messages, options={"temprature":0,"top_p": 0.9,"num_predict": 512,"repeat_penalty": 1.1})["message"]["content"]

# Decision Agent
def get_decision(user_query, context, available_files):
    decision_prompt = f"""
You are a code exploration agent. Your goal is to answer the user's question in the fewest steps possible.

Available tools:
- "search" for relevant code snippets
- "read" to load an entire file (from the provided list)
- "answer" when you have enough information

Rules:
- Max 2 search actions before answering
- No repeated searches or file reads
- Use only files from the provided list
- Respond with JSON: {{"action": "search/read/answer", "args": "<keywords/file path>", "reasoning": "<why you chose this>"}}

File List: {json.dumps(available_files)}
Question: {user_query}
Current Context: {context}
"""
    raw_response = ollama_chat(model=OLLAMA_MODEL, messages=[
        {"role": "system", "content": "You are a structured code exploration agent."},
        {"role": "user", "content": decision_prompt}
    ]).strip()
    json_text = extract_json(raw_response)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        print("Invalid JSON from LLM. Raw response:", raw_response)
        time.sleep(1)
        return get_decision(user_query, context, available_files)

# Main Agent Loop
def ask_agent(user_query, files, max_steps=15):
    context = ""
    seen_files = set()
    trace = []
    print("Question asked:", user_query)
    for step in range(max_steps):
        decision = get_decision(user_query, context, files)
        action = decision.get("action")
        args = decision.get("args", "")
        reasoning = decision.get("reasoning", "")
        trace.append({"step": step+1, "action": action, "args": args, "reasoning": reasoning})
        print(f"Step {step+1}: {decision}")
        if action == "answer":
            final_prompt = f"""
Answer the question based on the following context:
Question: {user_query}
Context:
{context}
"""
            answer = ollama_chat(model=OLLAMA_MODEL, messages=[
                {"role": "system", "content": "You are an expert code assistant."},
                {"role": "user", "content": final_prompt}
            ])
            return {"answer": answer, "trace": trace}
        elif action == "search":
            results = search_code(args)
            for file, snippet in results:
                if file not in seen_files:
                    context += f"\n# From {file}\n{snippet}\n"
                    seen_files.add(file)
        elif action == "read":
            if args in files and args not in seen_files:
                file_content = read_file(args)
                if file_content:
                    context += f"\n# Full file: {args}\n{file_content}\n"
                    seen_files.add(args)
            else:
                print(f"Agent requested invalid or already-read file: {args}")
    return {"answer": "Reached max steps without a confident answer.", "trace": trace}

# CLI Setup
def main():
    parser = argparse.ArgumentParser(description="Codebase Query System")
    parser.add_argument('--path', type=str, required=True, help='Path to the code directory')
    parser.add_argument('--question', type=str, required=True, help='Question to ask about the codebase')

    args = parser.parse_args()

    files = list_files(args.path)
    if len(collection.get()['ids']) == 0:
        index_codebase(args.path)

    result = ask_agent(args.question, files)
    print("\n=== FINAL ANSWER ===\n", result["answer"])
    print("\n=== AGENT TRACE ===\n", json.dumps(result["trace"], indent=2))

if __name__ == "__main__":
    main()
