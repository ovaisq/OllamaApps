#!/usr/bin/env python3

"""
Markdown Chatbot using Ollama, ChromaDB, and Gradio.

This script implements a chatbot that uses local LLMs (via Ollama) to answer questions based on a knowledge base stored in ChromaDB.
It supports conversation history, context retrieval, and real-time streaming of responses.
"""

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
RELOAD_INTERVAL_SECONDS = 30  # seconds
# ==================

logging.getLogger("chromadb").setLevel(logging.ERROR)

client = ollama.Client(host=OLLAMA_HOST)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name=CHROMA_COLLECTION)


def stop_chat(chat_history, history_state):
    """
    Stops the chat session by clearing input and returning current state.

    Args:
        chat_history (list): List of message dictionaries.
        history_state (list): Internal conversation history list.

    Returns:
        tuple: Updated chat history, empty input string, and history state.
    """
    return chat_history, "", history_state


def retrieve_context(query, k=TOP_K):
    """
    Retrieves the most relevant documents from ChromaDB based on a query embedding.

    Args:
        query (str): User's query text.
        k (int): Number of top results to retrieve.

    Returns:
        list: List of document texts retrieved as context.
    """
    query_embedding = client.embeddings(model=EMBEDD_MODEL, prompt=query)["embedding"]
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results["documents"][0]


def get_last_conversation(history, pairs=3):
    """
    Extracts the last few user-assistant conversation pairs from chat history.

    Args:
        history (list): Full chat history list.
        pairs (int): Maximum number of pairs to extract.

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


def get_answer(query, history):
    """
    Generates a response to the user's query using context and conversation history.

    Args:
        query (str): User’s input question.
        history (list): Conversation history.

    Yields:
        str: Streaming tokens of the generated answer.
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
        token = chunk.get("message", {}).get("content", "")
        answer += token
        yield answer
    return answer


def respond(message, chat_history, history_state):
    """
    Handles a new message from the user and generates a response.

    Args:
        message (str): User input.
        chat_history (list): Current chat messages.
        history_state (list): Internal state of conversation.

    Yields:
        tuple: Updated chat history, empty input string, and updated history state.
    """
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
    """
    Periodically reloads the ChromaDB collection in the background to reflect any changes.

    Args:
        interval_seconds (int): Interval between reload attempts.
    """
    global collection
    while True:
        try:
            collection = chroma_client.get_collection(name=CHROMA_COLLECTION)
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
