#!/usr/bin/env python3
"""Codebase indexer and query agent using embeddings and LLM."""

import argparse
import concurrent.futures
import glob
import json
import os
import re
import time
from typing import Optional

import chromadb
import ollama


OLLAMA_MODEL = "phi4-mini"
EMBEDDING_MODEL = "granite-embedding:278m"
MAX_SEARCH_STEPS = 2
CONTEXT_CHUNK_SIZE = 500
MAX_RESPONSE_TOKENS = 512


def _setup_chromadb() -> chromadb.Collection:
    client = chromadb.Client()
    return client.get_or_create_collection("codebase")


def _extract_json(raw_text: str) -> str:
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    return match.group(1) if match else raw_text


def _embed_text(text: str) -> Optional[list]:
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        return response.get("embedding")
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def _process_chunk(collection, file_path: str, idx: int, chunk: str) -> None:
    embedding = _embed_text(chunk)
    if embedding:
        collection.add(
            ids=[f"{file_path}_{idx}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"file": file_path}],
        )


def _is_excluded(file_path: str) -> bool:
    excluded_ext = {'.png', '.pem', '.jpg', '.log', '.yml', '.pyc', '.p8',
                    '.gif', '.woff', '.eot', '.tiff', '.ttf', '.webp', '.svg',
                    '.jpeg', '.ico', '.mp3'}
    excluded_names = {'Dockerfile'}
    ext = os.path.splitext(file_path)[1].lower()
    name = os.path.basename(file_path)
    return ext in excluded_ext or name in excluded_names


def _index_codebase(path: str, collection) -> list:
    files_indexed = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for file_path in _glob_files(path):
            if _is_excluded(file_path):
                continue
            content = _read_file(file_path)
            if not content:
                continue
            chunks = [content[i:i + CONTEXT_CHUNK_SIZE]
                     for i in range(0, len(content), CONTEXT_CHUNK_SIZE)]
            for idx, chunk in enumerate(chunks):
                futures.append(executor.submit(_process_chunk, collection, file_path, idx, chunk))
            files_indexed.append(file_path)
        concurrent.futures.wait(futures)
    print(f"Indexed {len(files_indexed)} files.")
    return files_indexed


def _glob_files(path: str) -> list:
    return glob.glob(f"{path}/**/*.*", recursive=True)


def _search_code(collection, query: str, k: int = 3) -> list:
    embedding = _embed_text(query)
    if not embedding:
        return []
    results = collection.query(query_embeddings=[embedding], n_results=k)
    docs = results["documents"][0]
    meta = results["metadatas"][0]
    return [(meta[i]["file"], docs[i]) for i in range(len(docs))]


def _read_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        print(f"Invalid path: {path}")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None


def _list_files(path: str) -> list:
    return [f for f in glob.glob(f"{path}/**/*.py", recursive=True)]


def _chat(model: str, messages: list, **options) -> str:
    response = ollama.chat(
        model=model,
        messages=messages,
        options=options
    )
    return response["message"]["content"]


def _get_decision(collection, user_query: str, context: str,
                 files: list) -> dict:
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

File List: {json.dumps(files)}
Question: {user_query}
Current Context: {context}
"""
    raw_response = _chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": "You are a structured code exploration agent."},
            {"role": "user", "content": decision_prompt}
        ],
        temperature=0,
        top_p=0.9,
        num_predict=MAX_RESPONSE_TOKENS,
        repeat_penalty=1.1
    ).strip()
    json_text = _extract_json(raw_response)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        print("Invalid JSON from LLM. Raw response:", raw_response)
        time.sleep(1)
        return _get_decision(collection, user_query, context, files)


def _ask_agent(collection, user_query: str, files: list,
              max_steps: int = 15) -> dict:
    context = ""
    seen_files = set()
    trace = []
    print("Question asked:", user_query)

    for step in range(max_steps):
        decision = _get_decision(collection, user_query, context, files)
        action = decision.get("action")
        args = decision.get("args", "")
        reasoning = decision.get("reasoning", "")
        trace.append({"step": step + 1, "action": action, "args": args, "reasoning": reasoning})
        print(f"Step {step + 1}: {decision}")

        if action == "answer":
            final_prompt = f"""
Answer the question based on the following context:
Question: {user_query}
Context:
{context}
"""
            answer = _chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert code assistant."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0,
                top_p=0.9,
                num_predict=MAX_RESPONSE_TOKENS,
                repeat_penalty=1.1
            )
            return {"answer": answer, "trace": trace}

        elif action == "search":
            results = _search_code(collection, args)
            for file, snippet in results:
                if file not in seen_files:
                    context += f"\n# From {file}\n{snippet}\n"
                    seen_files.add(file)

        elif action == "read":
            if args in files and args not in seen_files:
                content = _read_file(args)
                if content:
                    context += f"\n# Full file: {args}\n{content}\n"
                    seen_files.add(args)
            else:
                print(f"Invalid or already-read file: {args}")

    return {"answer": "Reached max steps without a confident answer.", "trace": trace}


def main() -> None:
    parser = argparse.ArgumentParser(description="Codebase Query System")
    parser.add_argument('--path', required=True, help='Path to the code directory')
    parser.add_argument('--question', required=True, help='Question about the codebase')
    parser.add_argument('--reindex', action='store_true', help='Force reindex')
    args = parser.parse_args()

    collection = _setup_chromadb()

    if args.reindex or len(collection.get()['ids']) == 0:
        _index_codebase(args.path, collection)

    files = _list_files(args.path)
    result = _ask_agent(collection, args.question, files)

    print("\nFinal Answer:\n", result["answer"])
    print("\nExecution Trace:\n", result["trace"])


if __name__ == "__main__":
    main()
