#!/usr/bin/env python3

import ollama
import chromadb
import gradio as gr
import threading
import time
import logging

# ===== CONFIG =====
CHAT_MODEL = 'phi4-mini'
EMBEDD_MODEL = 'snowflake-arctic-embed'
CHROMA_COLLECTION = "readme_rag"
CHROMA_DB_PATH = "./chroma_db"
OLLAMA_HOST = "http://"
TOP_K = 3
RELOAD_INTERVAL_SECONDS = 30  #seconds
# ==================

logging.getLogger("chromadb").setLevel(logging.ERROR)

client = ollama.Client(host=OLLAMA_HOST)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name=CHROMA_COLLECTION)

def stop_chat(chat_history, history_state):
    # Clear the input textbox and return current state to UI
    return chat_history, "", history_state

def retrieve_context(query, k=TOP_K):
    query_embedding = client.embeddings(model=EMBEDD_MODEL, prompt=query)["embedding"]
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results["documents"][0]

def get_last_conversation(history, pairs=3):
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

def get_answer(query, history):
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
        token = chunk.get("message", {}).get("content", "")
        answer += token
        yield answer
    return answer

def respond(message, chat_history, history_state):
    chat_history = chat_history or []
    response_gen = get_answer(message, chat_history)
    partial = ""
    for chunk in response_gen:
        partial = chunk
        yield chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": partial}
        ], "", history_state
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": partial})
    yield chat_history, "", history_state

def background_reloader(interval_seconds=RELOAD_INTERVAL_SECONDS):
    global collection
    while True:
        try:
            collection = chroma_client.get_collection(name=CHROMA_COLLECTION)
            # Uncomment below to log reloads:
            print("[INFO] Index reloaded in background.")
        except Exception as e:
            print(f"[ERROR] Failed to reload index: {e}")
        time.sleep(interval_seconds)

# Start background reload thread
reload_thread = threading.Thread(target=background_reloader, daemon=True)
reload_thread.start()

with gr.Blocks(title="Markdown Chatbot", css='footer {display: none !important;}') as chat:
    gr.Markdown("# ChromaDB: Markdown Chatbot")
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(label="Ask about the README")
    stop_btn = gr.Button("Stop Chat")
    history_state = gr.State([])

    msg.submit(respond, [msg, chatbot, history_state], [chatbot, msg, history_state], queue=True)
    stop_btn.click(stop_chat, [chatbot, history_state], [chatbot, msg, history_state])

if __name__ == "__main__":
    chat.queue().launch(server_name='0.0.0.0')

