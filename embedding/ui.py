#!/usr/bin/env python3
"""Chat Ollama Web Assistant

    This application leverages Gradio to create an interactive user interface
    for querying a set of web documents and generating responses based on their
    content. It integrates with OpenAI's language models, PostgreSQL for data
    storage and retrieval, and vector embeddings to enhance the search
    capabilities. The app also utilizes custom modules like `duckduckgo_search` to
    perform DuckDuckGo searches and generate HTML lists.

    Key Features:
        - Accepts multiple URLs as input.
        - Processes the text from these URLs using NLP techniques to embed them
            into vectors.
        - Stores these vectors in a PostgreSQL database for efficient querying.
        - Allows users to enter questions or instructions that are then used to
            query the embedded document content.
        - Generates responses based on the queried information, appending
            topic-relevant keywords.
        - Uses DuckDuckGo to search for keyphrases extracted from processed text
            and generate HTML ordered lists.

    User Interface:
        - Textbox for entering multiple newline-separated URLs.
        - Textbox for inputting questions or instructions.
        - Buttons and outputs for displaying the processed results, keyword list,
            and HTML content.

    Execution:
        Run this script as the main program, launching it on a server with the
        option for PWA (Progressive Web App) support.
"""

import json
import time
import os
import random
import re
import traceback
import sys
from typing import List

# Set USER_AGENT before other imports
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
]
os.environ['USER_AGENT'] = random.choice(user_agents)

import gradio as gr

import psycopg2
from psycopg2 import sql

# Check Python version and warn about compatibility
python_version = sys.version_info
if python_version.major == 3 and python_version.minor >= 14:
    print(f"WARNING: Running on Python {python_version.major}.{python_version.minor}")
    print("Some langchain components may have compatibility issues.")
    print("Consider using Python 3.11 or 3.12 for best compatibility.")

try:
    import openlit
    openlit_available = True
except ImportError:
    print("WARNING: openlit not available, monitoring disabled")
    openlit_available = False

try:
    from langchain_core.documents import Document
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.prompts import ChatPromptTemplate
    langchain_available = True
except ImportError as e:
    print(f"WARNING: LangChain import error: {e}")
    langchain_available = False

try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    ollama_available = True
except ImportError as e:
    print(f"WARNING: langchain_ollama import error: {e}")
    print("Ollama features will be disabled. Please update langchain packages:")
    print("  pip install --upgrade langchain-ollama langchain-core")
    ollama_available = False

try:
    from langchain_postgres import PGVector
    pgvector_available = True
except ImportError:
    print("WARNING: langchain_postgres not available")
    pgvector_available = False

from config import get_config
from websearch import create_dict_list_from_text, get_web_results_as_html

# Configuration
CONFIG = get_config()

host = CONFIG.get('psqldb', 'host')
database = CONFIG.get('psqldb', 'database')
user = CONFIG.get('psqldb', 'user')
password = CONFIG.get('psqldb', 'password')
port = CONFIG.get('psqldb', 'port')

OLLAMA_HOST = CONFIG.get('ai', 'OLLAMA_HOST')
LLM = CONFIG.get('ai', 'LLM') or "llama3.2"
EMBED_MODEL = "nomic-embed-text"

SERVICE_VERSION = CONFIG.get('service', 'version')

PGVECTOR_CONNECTION = f"postgresql+psycopg://{user}:{password}@{host}:5432/{database}"

PG_CONN_PARAMS = {
    'dbname': database,
    'user': user,
    'password': password,
    'host': host,  # e.g., 'localhost' or an IP address
    'port': port   # default is usually 5432
}

if OLLAMA_HOST:
    os.environ['OLLAMA_HOST'] = OLLAMA_HOST
else:
    os.environ['OLLAMA_HOST'] = ""

# Initialize OpenLIT if available
if openlit_available:
    try:
        openlit.init(
            otlp_endpoint=CONFIG.get('otlp','OTLP_ENDPOINT_URL'),
            collect_gpu_stats=CONFIG.get('otlp','COLLECT_GPU_STATS'),
            pricing_json=CONFIG.get('otlp','PRICING_JSON'),
            environment='production',
            application_name='ollama-web-assistant'
        )
    except Exception as e:
        print(f"WARNING: OpenLIT initialization failed: {e}")

def set_random_user_agent():
    """Picks a random user-agent from a given list and sets it as an environment variable USER_AGENT.

        Returns:
            str: The randomly selected user-agent string, or None if the list is empty.
    """

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Mobile Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
        "Mozilla/5.0 (Android 13; Mobile; rv:131.0) Gecko/20100101 Firefox/131.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/16F73 Safari/604.1",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 OPR/136.0.0.0",
        "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko"
    ]

    random_user_agent = random.choice(user_agents)
    os.environ['USER_AGENT'] = random_user_agent

def get_keyphrase_trends():
    """Get list of keyphrases"""

    try:
        # Establish a connection to the database using the provided configuration
        conn = psycopg2.connect(**PG_CONN_PARAMS)

        # Create a cursor object
        cur = conn.cursor()

        # Execute the query with the provided parameters
        sql_q = """
        WITH ExtractedLines AS (
            SELECT
                jsonb_array_elements_text(summarized_results->'urls') AS urls,
                summarized_results->>'new_results' AS new_results, -- Include new_results
                regexp_replace(line_text, '^\\d+\\.\\s*', '', 'g') AS line_text, -- Remove leading numbers
                ordinality
            FROM
                summarized_results,
                LATERAL unnest(string_to_array(summarized_results->>'keyword_list', E'\\n')) WITH ORDINALITY AS t(line_text, ordinality)
        ),
        FilteredLines AS (
            SELECT urls, new_results, line_text, ordinality
            FROM ExtractedLines
            WHERE ordinality > (
                SELECT MAX(ordinality) - 3
                FROM ExtractedLines e2
                WHERE ExtractedLines.urls = e2.urls
            )
        )
        SELECT line_text AS keyword_list
        FROM FilteredLines
        WHERE line_text != ''
        ORDER BY ordinality;
        """
        cur.execute(sql_q)

        # Fetch all results
        rows = cur.fetchall()
        blob_of_text = "\n".join([row[0] for row in rows])
        # Close communication with the database
        cur.close()
        conn.close()

        # Return True if any rows were found, otherwise False
        return blob_of_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def check_for_url_and_query(url_to_search, keyword_pattern):
    """Checks if there exists a row in the summarized_results table that matches the specified
        criteria.
    """

    try:
        # Establish a connection to the database using the provided configuration
        conn = psycopg2.connect(**PG_CONN_PARAMS)

        # Create a cursor object
        cur = conn.cursor()

        # Execute the query with the provided parameters
        cur.execute( """
                SELECT *
                FROM summarized_results
                WHERE summarized_results->'urls' ? %s
                AND lower((summarized_results->>'q_n_i')) LIKE %s;
                """, (url_to_search, (keyword_pattern.lower() + '%')))

        # Fetch all results
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        query_results = [dict(zip(columns, row)) for row in rows]
        # Close communication with the database
        cur.close()
        conn.close()

        # Return True if any rows were found, otherwise False
        return query_results

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def insert_json_to_table(json_doc):
    """Insert JSON into JSONB column"""

    try:
        # Establish a connection to the database using the provided configuration
        conn = psycopg2.connect(**PG_CONN_PARAMS)

        # Create a cursor object
        cur = conn.cursor()

        # Parse JSON document
        json_blob = json.loads(json_doc)

        # Define SQL query with a placeholder for the JSONB column
        query_p = sql.SQL("INSERT INTO summarized_results (summarized_results) VALUES (%s)")

        # Execute the query, passing the JSON as a parameter
        cur.execute(query_p, (json.dumps(json_blob),))

        # Commit the transaction
        conn.commit()

        # Close the cursor and connection
        cur.close()
        conn.close()

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def extract_section(text):
    """Extract the 'Topic-Relevant Keywords:' section from text"""
    match = re.search(r"Topic-Relevant Keywords:(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def remove_section(text):
    """Remove the 'Topic-Relevant Keywords:' section from text"""
    return re.sub(r"Topic-Relevant Keywords:.*", "", text, flags=re.DOTALL).strip()

def load_documents(url_list: List[str]) -> List:
    """Loads documents from a list of URLs."""

    if not langchain_available:
        return []

    try:
        docs = [WebBaseLoader(url).load() for url in url_list]
        return [item for sublist in docs for item in sublist]
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def embed_and_store_documents(documents: List):
    """Embeds documents and stores them in PGVector."""

    if not (ollama_available and pgvector_available and langchain_available):
        print("Required packages not available for embedding")
        return

    try:
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500,
                                                                    chunk_overlap=100)
        doc_splits = text_splitter.split_documents(documents)

        embedding_model = OllamaEmbeddings(model=EMBED_MODEL)
        vectorstore = PGVector.from_documents(doc_splits, embedding_model,
                                                connection=PGVECTOR_CONNECTION)

        with psycopg2.connect(**PG_CONN_PARAMS) as conn:
            with conn.cursor() as cur:
                for doc in doc_splits:
                    cur.execute("SELECT id FROM rag_pgvector WHERE document = %s",
                                (doc.page_content,))
                    if not cur.fetchone():
                        if langchain_available:
                            vectorstore.add_documents([Document(page_content=doc.page_content)])
    except Exception as e:
        print(f"Error embedding documents: {e}\n{traceback.format_exc()}")

def query_documents(url_list: List[str], query: str) -> str:
    """Queries the documents for the given question or instruction."""

    if not (ollama_available and langchain_available):
        return "Error: Required langchain packages not available. Please upgrade:\n\npip install --upgrade langchain-ollama langchain-core langchain-community"

    try:
        # Load and process documents
        documents = load_documents(url_list)
        if not documents:
            return "No documents found or failed to load documents."

        embed_and_store_documents(documents)

        # Combine all document content into a single context string
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500,
                                                                    chunk_overlap=100)
        doc_splits = text_splitter.split_documents(documents)
        retriever_context = "\n".join([doc.page_content for doc in doc_splits])

        # Define prompt template
        prompt_template = """
        Based on the following context, respond to the query:
        Context: {context}
        Query: {query}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Initialize the model
        model = ChatOllama(model=LLM, temperature=0.7)

        append_query = (
            "Additionally, list 3 topic-relevant keywords as a numbered "
            "list at the end. Always label the list with exact same title "
            "'Topic-Relevant Keywords:'. This must be a list of three nothing else."
        )

        # Create the input for the pipeline
        input_data = {"context": retriever_context, "query": query + append_query}

        # Run the pipeline
        prompt_result = prompt.invoke(input_data)  # Process input through the prompt
        model_result = model.invoke(prompt_result)  # Process the prompt result through the model
        return model_result.content if hasattr(model_result, "content") else "No content available."
    except Exception as e:
        print(f"Error querying documents: {e}\n{traceback.format_exc()}")
        return f"An error occurred while processing the query: {str(e)}"

def get_trend_summary():

    if not (ollama_available and langchain_available):
        return "⚠️ Trending topics unavailable (langchain packages need update)"

    try:
        # Define prompt template
        prompt_template = """
        Based on the following context, respond to the query:
        Context: {context}
        Query: {query}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        my_context = get_keyphrase_trends()
        
        if not my_context:
            return "📊 No trending data available yet. Start querying to build trends!"

        # Initialize the model
        model = ChatOllama(model=LLM, temperature=0.7)

        # Create the input for the pipeline
        input_data = {"context": my_context, "query": "Based on this come up with trending topics, respond as a table of 3 columns. Nothing before or after or outside of the table."}

        # Run the pipeline
        prompt_result = prompt.invoke(input_data)  # Process input through the prompt
        model_result = model.invoke(prompt_result)  # Process the prompt result through the model
        return model_result.content if hasattr(model_result, "content") else "No content available."
    except Exception as e:
        print(f"Error querying documents: {e}\n{traceback.format_exc()}")
        return "⚠️ An error occurred while getting trends."

def process_input(urls_str: str, q_n_i: str):
    """Processes the input URLs and query to generate a response."""

    set_random_user_agent()

    urls_list = urls_str.strip().split("\n")
    if not urls_list or not q_n_i.strip():
        return "⚠️ Please provide both URL(s) and a query.", "", ""

    json_doc = ''

    lookup_existing_results = check_for_url_and_query(urls_list[0], q_n_i)

    if lookup_existing_results:
        lookup_existing_results = lookup_existing_results[0]['summarized_results']

    if not lookup_existing_results:
        orig_results = query_documents(urls_list, q_n_i) + '\n'
        keyword_list = extract_section(orig_results)
        new_results = remove_section(orig_results)
        # Generate the list of dictionaries from the sample text
        dicts = create_dict_list_from_text(keyword_list)
        # Generate and print the HTML ordered list
        web_results_html = get_web_results_as_html(dicts)

        json_doc = {
                    'timestamp' : int(time.time()),
                    'urls' : urls_list,
                    'q_n_i' : q_n_i,
                    'new_results' : new_results,
                    'keyword_list' : keyword_list,
                    'web_results_html' : web_results_html
                   }
        insert_json_to_table(json.dumps(json_doc))
    else:
        new_results = lookup_existing_results['new_results']
        keyword_list = lookup_existing_results['keyword_list']
        web_results_html = lookup_existing_results['web_results_html']

    return new_results, keyword_list, web_results_html


# Modern Glassmorphic CSS inspired by foo.html
custom_css = """
    /* CSS Variables */
    :root {
        --primary-bg: #0f172a;
        --glass-bg: rgba(30, 41, 59, 0.7);
        --glass-border: rgba(255, 255, 255, 0.1);
        --accent-color: #4A60A1;
        --accent-hover: #5a7ac4;
        --text-main: #e2e8f0;
        --text-muted: #94a3b8;
        --success: #10b981;
        --error: #ef4444;
        --font-stack: 'Segoe UI', system-ui, -apple-system, sans-serif;
        --card-radius: 12px;
    }

    /* Global Body Styling */
    body, .gradio-container {
        font-family: var(--font-stack) !important;
        background-color: var(--primary-bg) !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(74, 96, 161, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(74, 96, 161, 0.15) 0px, transparent 50%) !important;
        background-attachment: fixed !important;
        color: var(--text-main) !important;
    }

    /* Card/Box Styling with Glassmorphism */
    #trends-box, #results-box, #keywords-box, #web-box {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--card-radius) !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(10px) !important;
        min-height: 150px !important;
        color: var(--text-main) !important;
    }

    /* Specific styling for trends */
    #trends-box {
        background: rgba(74, 96, 161, 0.2) !important;
        border: 1px solid rgba(74, 96, 161, 0.3) !important;
    }

    /* Keywords styling */
    #keywords-box {
        background: rgba(74, 96, 161, 0.25) !important;
    }

    /* Web results with darker background */
    #web-box {
        background: rgba(0, 0, 17, 0.8) !important;
        color: #a5b4fc !important;
    }

    #web-box a {
        color: #a5b4fc !important;
        text-decoration: none;
    }

    #web-box a:hover {
        color: #c7d2fe !important;
        text-decoration: underline;
    }

    /* Textbox Styling */
    .gradio-container textarea,
    .gradio-container input[type="text"] {
        background: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid var(--glass-border) !important;
        color: var(--text-main) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        transition: border-color 0.2s !important;
    }

    .gradio-container textarea:focus,
    .gradio-container input[type="text"]:focus {
        border-color: var(--accent-color) !important;
        outline: none !important;
    }

    /* Button Styling */
    .gradio-container button.primary {
        background: var(--accent-color) !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: transform 0.1s, background-color 0.2s !important;
    }

    .gradio-container button.primary:hover {
        background: var(--accent-hover) !important;
    }

    .gradio-container button.primary:active {
        transform: scale(0.98) !important;
    }

    /* Label Styling */
    .gradio-container label {
        color: var(--text-muted) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* Header/Title Styling */
    .gradio-container h1, .gradio-container h2, .gradio-container h3 {
        background: linear-gradient(90deg, #fff, #94a3b8) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }

    /* Markdown content styling */
    .gradio-container .markdown {
        color: var(--text-main) !important;
    }

    .gradio-container .markdown strong {
        color: #fff !important;
    }

    .gradio-container .markdown code {
        background: rgba(255, 255, 255, 0.1) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        color: #a5b4fc !important;
    }

    /* Copy button styling */
    .gradio-container button[title="Copy"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid var(--glass-border) !important;
        color: var(--text-main) !important;
        border-radius: 6px !important;
        transition: all 0.2s !important;
    }

    .gradio-container button[title="Copy"]:hover {
        background: rgba(255,255,255,0.1) !important;
        border-color: var(--accent-color) !important;
    }

    /* Animate trending pills */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    #trends-box .markdown {
        animation: fadeIn 0.5s ease-out;
    }

    /* Footer styling */
    .gradio-container footer {
        color: var(--text-muted) !important;
    }

    /* Status/Progress indicator */
    .gradio-container .progress-bar {
        background: var(--accent-color) !important;
    }

    /* Warning/Error message styling */
    .gradio-container .error, .gradio-container .warning {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        color: #fca5a5 !important;
        padding: 1rem !important;
        border-radius: 8px !important;
    }
"""

# Define Gradio UI with modern styling (Gradio 6.0 compatible)
with gr.Blocks() as ui:
    # System status message
    status_parts = []
    if not ollama_available:
        status_parts.append("⚠️ langchain-ollama unavailable")
    if not langchain_available:
        status_parts.append("⚠️ langchain unavailable")
    if python_version.major == 3 and python_version.minor >= 14:
        status_parts.append(f"⚠️ Python {python_version.major}.{python_version.minor} - compatibility issues possible")
    
    if status_parts:
        gr.Markdown(f"### System Status\n{' | '.join(status_parts)}\n\n**Recommended:** Use Python 3.11 or 3.12 and run:\n```\npip install --upgrade langchain-ollama langchain-core langchain-community\n```")
    
    gr.Markdown("# 🚀 Ollama Web Assistant")
    gr.Markdown("### 📊 Trending Topics")

    with gr.Row():
        # Removed show_copy_button parameter (not available in Gradio 6.0 for Markdown)
        trends = gr.Markdown(label="Current Trends", elem_id='trends-box', every=15)
    ui.load(get_trend_summary, inputs=None, outputs=trends)

    gr.Markdown("### 🔍 Query Interface")

    with gr.Row():
        with gr.Column(scale=1):
            urls = gr.Textbox(
                label="📎 Enter URL(s)", 
                placeholder="https://example.com\nhttps://another-example.com",
                lines=3
            )
        with gr.Column(scale=2):
            q_n_a = gr.Textbox(
                label="❓ Ask a question or provide an instruction",
                placeholder="What are the key insights from these documents?",
                lines=3
            )

    with gr.Row():
        submit_button = gr.Button("🔄 Process Query", variant="primary")

    gr.Markdown("### 📋 Results")

    with gr.Row():
        with gr.Column():
            results = gr.Markdown(
                "Response will appear here...",
                elem_id="results-box", 
                label="💬 LLM Response"
            )
        with gr.Column():
            keywords = gr.Markdown(
                "Keywords will appear here...",
                elem_id="keywords-box", 
                label="🔑 Topic Keywords"
            )

    with gr.Row():
        html_list = gr.Markdown(
            "Web results will appear here...",
            elem_id="web-box",
            label="🌐 Related Web Content"
        )

    submit_button.click(
        fn=process_input, 
        inputs=[urls, q_n_a],
        outputs=[results, keywords, html_list]
    )
    
    gr.Markdown(
        f"""
        <div style='text-align: center; margin-top: 2rem; padding: 1rem; 
                    background: var(--glass-bg); border: 1px solid var(--glass-border); 
                    border-radius: 8px; color: var(--text-muted);'>
            <b>🤖 LLM</b>: {LLM} | <b>📊 Embeddings</b>: {EMBED_MODEL} | <b>📌 Version</b>: {SERVICE_VERSION} | <b>🐍 Python</b>: {python_version.major}.{python_version.minor}
        </div>
        """
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Ollama Web Assistant")
    print("="*60)
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print(f"LangChain Available: {langchain_available}")
    print(f"Ollama Available: {ollama_available}")
    print(f"PGVector Available: {pgvector_available}")
    print(f"OpenLIT Available: {openlit_available}")
    print("="*60 + "\n")
    
    # Gradio 6.0: Pass theme and css to launch() instead of Blocks()
    ui.launch(
        server_name="0.0.0.0",
        css=custom_css,
        theme=gr.themes.Glass()
    )
