#!/usr/bin/env python3

"""
A chatbot application that uses PostgreSQL vector search to retrieve context,
then feeds it into an LLM for answering questions about markdown content.

This application connects to a PostgreSQL database with vector embeddings,
retrieves relevant document chunks using approximate nearest neighbor search,
and generates responses using Ollama's language models.
"""

import ollama
import psycopg2
import gradio as gr
import threading
import time
import logging
import uuid
from psycopg2.extras import Json
import queue
from concurrent.futures import ThreadPoolExecutor

# ===== CONFIG =====
CHAT_MODEL = 'phi4-mini'
EMBEDD_MODEL = 'snowflake-arctic-embed'
DB_NAME = ""
DB_USER = ""
DB_PASSWORD = ""
DB_HOST = ""
DB_PORT = "5432"
TABLE_NAME = "markdown_chunks"
TOP_K = 3
RELOAD_INTERVAL_SECONDS = 30  # seconds
# ==================

logging.getLogger("psycopg2").setLevel(logging.ERROR)

client = ollama.Client(host="http://")

# Connect to PostgreSQL database
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cursor = conn.cursor()

# Global flag for stopping generation
stop_flag = threading.Event()

def retrieve_context(query, k=TOP_K):
    """
    Retrieve top-k relevant document chunks based on query embedding.
    
    Args:
        query (str): User's question to find relevant context for.
        k (int): Number of top matching chunks to return. Default is TOP_K.
        
    Returns:
        list: List of strings representing relevant document chunks.
    """
    query_embedding = client.embeddings(model=EMBEDD_MODEL, prompt=query)["embedding"]

    # Convert embedding list to PostgreSQL array format
    embedding_str = ",".join(map(str, query_embedding))
    cursor.execute(f"""
        SELECT chunk, metadata
        FROM {TABLE_NAME}
        ORDER BY embedding <=> '[{embedding_str}]'::vector
        LIMIT %s
    """, (k,))

    results = cursor.fetchall()
    documents = [row[0] for row in results]
    return documents

def get_last_conversation(history, pairs=3):
    """
    Extract recent conversation history from chat history.
    
    Args:
        history (list): Full chat history as list of message dicts.
        pairs (int): Maximum number of user-assistant pairs to extract.
        
    Returns:
        list: List of tuples containing (user_message, assistant_response).
    """
    conv = []
    i = len(history) - 1
    while i > 0 and len(conv) < pairs:
        if history[i]['role'] == 'assistant' and history[i-1]['role'] == 'user':
            conv.append((history[i-1]['content'], history[i]['content']))
            i -= 2
        else:
            i -= 1
    conv.reverse()  # chronological order
    return conv

def get_answer_streaming(query, history):
    """
    Generate a streaming response using Ollama's chat API.
    
    Yields:
        str: Partial responses as they are generated (streaming).
    """
    context_chunks = retrieve_context(query)
    context_text = "\n".join(context_chunks)
    conversation_context = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in get_last_conversation(history)])

    prompt = f"""
You are a helpful assistant. Use the following context to answer the question accurately.
Context:
{context_text}

Previous conversation:
{conversation_context}

Question: {query}
Answer:
"""

    stream = client.chat(model=CHAT_MODEL, options=dict(num_ctx=8192), messages=[{"role": "user", "content": prompt}], stream=True)
    answer = ""
    for chunk in stream:
        if stop_flag.is_set():
            break
        token = chunk.get("message", {}).get("content", "")
        answer += token
        yield answer

def respond(message, chat_history, history_state):
    """
    Handle user message submission and generate response.
    """
    global stop_flag
    stop_flag.clear()  # Reset the stop flag before starting a new generation

    chat_history = chat_history or []
    response_gen = get_answer_streaming(message, chat_history)
    partial = ""
    
    try:
        for chunk in response_gen:
            if stop_flag.is_set():
                break
            partial = chunk
            yield chat_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": partial}
            ], "", history_state
    except Exception as e:
        print(f"[ERROR] Streaming error: {e}")
    finally:
        # Final update after processing or interruption
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": partial})
        yield chat_history, "", history_state

def stop_chat(chat_history, history_state):
    """
    Stop ongoing chat generation and clear input textbox.
    """
    global stop_flag
    stop_flag.set()  # Signal streaming to stop
    return chat_history, "", history_state

def background_reloader(interval_seconds=RELOAD_INTERVAL_SECONDS):
    """
    Periodically reload database connection in background thread to prevent timeouts.
    
    Args:
        interval_seconds (int): Time interval between reload attempts. Default is RELOAD_INTERVAL_SECONDS.
    """
    global cursor, conn
    while True:
        try:
            # Refresh the connection (or cursor) if needed
            cursor.close()
            conn.close()

            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            cursor = conn.cursor()
            print("[INFO] Database connection reloaded in background.")
        except Exception as e:
            print(f"[ERROR] Failed to reload database connection: {e}")
        time.sleep(interval_seconds)

# Start background reload thread
reload_thread = threading.Thread(target=background_reloader, daemon=True)
reload_thread.start()

with gr.Blocks(title="PGVECTOR: Markdown Chatbot", css='footer {display: none !important;}') as chat:
    gr.Markdown("# PGVECTOR: Chatbot")
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(label="Ask about the README")
    stop_btn = gr.Button("Stop Chat")
    history_state = gr.State([])

    msg.submit(respond, [msg, chatbot, history_state], [chatbot, msg, history_state], queue=True)
    stop_btn.click(stop_chat, [chatbot, history_state], [chatbot, msg, history_state])

if __name__ == "__main__":
    chat.queue().launch(server_name='0.0.0.0', pwa=True)
