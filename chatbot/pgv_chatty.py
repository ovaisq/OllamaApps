#!/usr/bin/env python3
import logging
import os
import threading
import time
from typing import List, Dict, Any
import gradio as gr
import psycopg2

from pgv_config import DB_CONFIG, CHAT_CONFIG, OLLAMA_CONFIG
from pgv_utils import normalize_text

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

class PGVectorChat:
    def __init__(self):
        self.db_connection = None
        self.stop_flag = threading.Event()
        self.setup_database()
        self.start_background_reloader()
    
    def setup_database(self):
        """Setup database connection."""
        self.db_connection = psycopg2.connect(**DB_CONFIG)
    
    def get_context_chunks(self, query: str) -> List[str]:
        """Get relevant chunks using vector similarity search."""
        with self.db_connection.cursor() as cursor:
            # Simplified vector search - in production you'd use proper vector similarity
            cursor.execute(
                "SELECT chunk FROM markdown_chunks "
                "ORDER BY LENGTH(chunk) DESC LIMIT %s",
                (CHAT_CONFIG['top_k'],)
            )
            return [row[0] for row in cursor.fetchall()]
    
    def get_last_conversation(self, history: List[Dict], count: int = 2) -> List[tuple[str, str]]:
        """Get last conversation turns."""
        result = []
        i = len(history) - 1
        while i >= 0 and len(result) < count:
            if history[i]['role'] == 'assistant' and i > 0 and history[i-1]['role'] == 'user':
                result.append((history[i-1]['content'], history[i]['content']))
                i -= 2
            else:
                i -= 1
        return result[::-1]
    
    def get_answer_stream(self, query: str, history: List[Dict]) -> str:
        """Generate streaming response."""
        # Get context chunks
        context_chunks = self.get_context_chunks(query)
        context_text = "\n".join(context_chunks)
        
        # Get conversation context
        conversation_context = "\n".join([
            f"User: {u}\nAssistant: {a}" 
            for u, a in self.get_last_conversation(history)
        ])
        
        # Build prompt
        prompt = f"""
You are a helpful assistant. Use the following context to answer the question accurately.
Context:
{context_text}

Previous conversation:
{conversation_context}

Question: {query}
Answer:
"""
        
        # Connect to Ollama and stream response
        import ollama
        client = ollama.Client(host=OLLAMA_CONFIG['host'])
        
        try:
            response_stream = client.chat(
                model=OLLAMA_CONFIG['chat_model'],
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={
                    "num_ctx": CHAT_CONFIG['max_context_length']
                }
            )
            
            full_response = ""
            for chunk in response_stream:
                if self.stop_flag.is_set():
                    break
                content = chunk.get("message", {}).get("content", "")
                full_response += content
                yield full_response
                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def respond(self, message: str, history: List[Dict], state: Any) -> tuple:
        """Handle chat response."""
        self.stop_flag.clear()
        
        # Initialize response stream
        response_stream = self.get_answer_stream(message, history)
        
        # Build updated history
        new_history = history + [{"role": "user", "content": message}]
        
        # Stream response
        try:
            for partial_response in response_stream:
                if self.stop_flag.is_set():
                    break
                yield (
                    new_history + [{"role": "assistant", "content": partial_response}],
                    "",  # Clear input
                    state
                )
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            yield (
                new_history + [{"role": "assistant", "content": error_msg}],
                "",
                state
            )
    
    def stop_chat(self, history: List[Dict], state: Any) -> tuple:
        """Stop current chat response."""
        self.stop_flag.set()
        return (history, "", state)
    
    def start_background_reloader(self):
        """Start background database reloader."""
        def reloader():
            while True:
                time.sleep(CHAT_CONFIG['reload_interval'])
                try:
                    if self.db_connection.closed:
                        self.setup_database()
                    logging.info("Database connection refreshed")
                except Exception as e:
                    logging.error(f"Failed to refresh database connection: {e}")
        
        reload_thread = threading.Thread(target=reloader, daemon=True)
        reload_thread.start()

def main():
    """Main function for chat interface."""
    logging.basicConfig(level=logging.INFO)
    
    # Create chat instance
    chat = PGVectorChat()
    
    # Define Gradio interface
    with gr.Blocks(title="Markdown Document Chatbot", css='footer {display: none !important;}', theme='JohnSmith9982/small_and_pretty') as chatty:
        gr.Markdown("# Markdown Document Chatbot")
        
        chatbot = gr.Chatbot(type="messages", label="Chat History")
        msg = gr.Textbox(label="Your Message")
        stop_btn = gr.Button("Stop Response")
        state = gr.State()
        
        # Event handlers
        submit_event = msg.submit(
            chat.respond, 
            [msg, chatbot, state], 
            [chatbot, msg, state],
            queue=True
        )
        stop_btn.click(
            chat.stop_chat,
            [chatbot, state],
            [chatbot, msg, state]
        )
        
        # Clear history
        clear_btn = gr.Button("Clear History")
        clear_btn.click(lambda: ([], "", None), None, [chatbot, msg, state])
    
    # Launch interface
    chatty.queue()
    chatty.launch(server_name='0.0.0.0', share=False, pwa=True)

if __name__ == "__main__":
    main()
