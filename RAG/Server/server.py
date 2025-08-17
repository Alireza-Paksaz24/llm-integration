from flask import Flask, request, jsonify, render_template
import os
import sys
import json
import datetime
import requests
import subprocess
from dotenv import load_dotenv
import threading
import queue
import time
from pathlib import Path

# Add RAG folder to path
sys.path.append('../')

root_dir = Path(__file__).parent.parent  # Gets the root directory relative to this file
sys.path.append(str(root_dir))

# Fix tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check if vector search is available
try:
    from vector_search import search_top_k, get_document_stats
    VECTOR_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Vector search not available: {e}")
    print("Please run 'python embedding.py' first to create embeddings.")
    VECTOR_SEARCH_AVAILABLE = False

load_dotenv()

# ----- .env config -----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
LLM_API_KEY = os.getenv("LLM_API_KEY", None)
LOG_FILE = "../query_log.txt"
# ------------------------

app = Flask(__name__)

# Global conversation history
conversation_history = []

def run_llm(prompt: str) -> str:
    if LLM_PROVIDER == "ollama":
        return run_ollama(prompt)
    elif LLM_PROVIDER == "lmstudio":
        return run_lmstudio(prompt)
    elif LLM_PROVIDER == "openai":
        return run_openai(prompt)
    elif LLM_PROVIDER == "gemini":
        return run_gemini(prompt)
    elif LLM_PROVIDER == "deepseek":
        return run_deepseek(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def run_ollama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", LLM_MODEL, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    return result.stdout.strip()


def run_lmstudio(prompt: str) -> str:
    """Use subprocess curl to communicate with LM Studio (more reliable than requests)"""
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }
    
    curl_command = [
        'curl', '-s',  # -s for silent mode
        f'{LLM_BASE_URL}/v1/chat/completions',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps(payload)
    ]
    
    try:
        result = subprocess.run(
            curl_command, 
            capture_output=True, 
            text=True, 
            encoding="utf-8",
            errors="replace",
            timeout=120
        )
        
        if result.returncode != 0:
            raise Exception(f"Curl command failed with return code {result.returncode}: {result.stderr}")
        
        if not result.stdout.strip():
            raise Exception("Empty response from LM Studio")
            
        try:
            response_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON response: {result.stdout}")
        
        if "choices" not in response_data or not response_data["choices"]:
            raise Exception(f"Unexpected response format: {response_data}")
            
        return response_data["choices"][0]["message"]["content"]
        
    except subprocess.TimeoutExpired:
        raise Exception("Request to LM Studio timed out")
    except Exception as e:
        raise Exception(f"LM Studio request failed: {e}")


def run_openai(prompt: str) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def run_gemini(prompt: str) -> str:
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent?key={LLM_API_KEY}",
        json={"contents": [{"parts": [{"text": prompt}]}]},
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def run_deepseek(prompt: str) -> str:
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def log_to_file(message: str):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as log_f:
            log_f.write(message + "\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")


def detect_language(text):
    """Simple language detection based on character sets"""
    # Persian/Farsi Unicode ranges
    persian_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
    # Arabic Unicode ranges (different from Persian)
    arabic_chars = sum(1 for char in text if '\u0750' <= char <= '\u077F' or '\uFB50' <= char <= '\uFDFF' or '\uFE70' <= char <= '\uFEFF')
    
    total_chars = len(text.strip())
    if total_chars == 0:
        return 'en'
    
    persian_ratio = persian_chars / total_chars
    arabic_ratio = arabic_chars / total_chars
    
    if persian_ratio > 0.3:
        return 'fa'  # Persian/Farsi
    elif arabic_ratio > 0.3:
        return 'ar'  # Arabic
    else:
        return 'en'  # Default to English for Latin scripts


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({'error': 'Empty query provided'}), 400
        
        if not VECTOR_SEARCH_AVAILABLE:
            return jsonify({'error': 'Vector search not available. Please run embedding.py first.'}), 500
        
        # Detect language for RTL/LTR
        detected_lang = detect_language(user_query)
        
        # Log the query
        timestamp = datetime.datetime.now().isoformat()
        log_to_file(f"[{timestamp}] [Query] {user_query}")
        
        # Extract keywords
        keyword_prompt = f"""
Extract only the main keywords or key phrases from the following user query for a vector search.
Return keywords separated by commas, no explanation.

Query: {user_query}
"""
        
        keywords = run_llm(keyword_prompt)
        log_to_file(f"[{timestamp}] [Search Keywords] {keywords}")
        
        # Search documents
        top_k_docs = search_top_k(keywords)
        
        # Log search results
        log_to_file(f"[{timestamp}] [Search Results]:")
        for doc in top_k_docs:
            log_to_file(f"- File: {doc['file']} (Chunk {doc['chunk_index']})")
            log_to_file(f"  Similarity: {doc['similarity']:.4f}")
            log_to_file(f"  Text: {doc['text'][:200]}...")
        
        # Build context from documents
        document_context = "\n\n".join([
            f"[Document: {doc['file']}, Chunk {doc['chunk_index']}, Similarity: {doc['similarity']:.3f}]\n{doc['text']}"
            for doc in top_k_docs
        ])
        
        # Build conversation history context
        history_context = ""
        if conversation_history:
            history_context = "\n\nPrevious conversation context:\n"
            for i, (prev_q, prev_a) in enumerate(conversation_history[-3:], 1):
                history_context += f"\nPrevious Q{i}: {prev_q}\nPrevious A{i}: {prev_a}\n"
        
        # Generate answer
        prompt = f"""
You are a knowledgeable assistant answering the user's question using the provided information.

Instructions:
- Use the document context as your primary source of information
- Consider the conversation history to maintain context and avoid repetition
- If the answer is in the documents, explain clearly in your own words
- If partially present, combine information logically
- If insufficient information, say so politely and suggest what might help
- Be concise, friendly, and conversational
- Reference specific documents when relevant

Current Question: {user_query}

Document Context:
{document_context}

{history_context}

Please provide your detailed answer:
"""
        
        answer = run_llm(prompt)
        
        # Store this exchange in conversation history
        conversation_history.append((user_query, answer))
        
        # Keep only last 5 exchanges
        if len(conversation_history) > 5:
            conversation_history[:] = conversation_history[-5:]
        
        log_to_file(f"[{timestamp}] [Answer] {answer}")
        
        return jsonify({
            'answer': answer,
            'keywords': keywords,
            'documents': top_k_docs,
            'language': detected_lang,
            'timestamp': timestamp
        })
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        log_to_file(f"[{datetime.datetime.now().isoformat()}] [Error] {error_msg}")
        return jsonify({'error': error_msg}), 500


@app.route('/api/logs')
def get_logs():
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                logs = f.readlines()
            # Return last 100 lines
            return jsonify({'logs': logs[-100:]})
        else:
            return jsonify({'logs': []})
    except Exception as e:
        return jsonify({'error': f"Error reading logs: {str(e)}"}), 500


@app.route('/api/stats')
def get_stats():
    try:
        if VECTOR_SEARCH_AVAILABLE:
            stats = get_document_stats()
            return jsonify(stats)
        else:
            return jsonify({'error': 'Vector search not available'})
    except Exception as e:
        return jsonify({'error': f"Error getting stats: {str(e)}"}), 500


@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    global conversation_history
    conversation_history.clear()
    return jsonify({'message': 'Conversation history cleared'})


if __name__ == '__main__':
    print(f"🚀 Starting RAG Server with {LLM_PROVIDER} ({LLM_MODEL})")
    print(f"📊 Vector search available: {VECTOR_SEARCH_AVAILABLE}")
    app.run(debug=True, host='0.0.0.0', port=5000)