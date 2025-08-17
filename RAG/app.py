import os
import json
import datetime
import requests
import subprocess
from dotenv import load_dotenv

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
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")   # ollama | lmstudio | openai | gemini | deepseek
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
LLM_API_KEY = os.getenv("LLM_API_KEY", None)
LOG_FILE = "query_log.txt"
# ------------------------

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
    with open(LOG_FILE, "a", encoding="utf-8") as log_f:
        log_f.write(message + "\n")


# Main application loop
conversation_history = []  # Store previous Q&A pairs

try:
    print(f"RAG System initialized with {LLM_PROVIDER} ({LLM_MODEL})")
    print("Ready for queries!")
    
    while True:
        query = input("\nEnter your query (or Ctrl+C to exit): ").strip()
        if not query:
            continue

        print("🔍 Extracting keywords...")
        keyword_prompt = f"""
Extract only the main keywords or key phrases from the following user query for a vector search.
Return keywords separated by commas, no explanation.

Query: {query}
"""
        keywords = run_llm(keyword_prompt)
        print(f"📝 Keywords extracted: {keywords}")
        log_to_file(f"[{datetime.datetime.now()}] [Query] {query}")
        log_to_file(f"[{datetime.datetime.now()}] [Search Keywords] {keywords}")

        print("📚 Searching documents...")
        top_k_docs = search_top_k(keywords)
        print(f"📄 Found {len(top_k_docs)} relevant document chunks")
        
        log_to_file(f"[{datetime.datetime.now()}] [Search Results]:")
        for doc in top_k_docs:
            log_to_file(f"- File: {doc['file']} (Chunk {doc['chunk_index']})")
            log_to_file(f"  Similarity: {doc['similarity']:.4f}")
            log_to_file(f"  Text: {doc['text']}\n")

        # Build context from documents
        document_context = "\n\n".join([
            f"[Document: {doc['file']}, Chunk {doc['chunk_index']}, Similarity: {doc['similarity']:.3f}]\n{doc['text']}"
            for doc in top_k_docs
        ])

        # Build conversation history context
        history_context = ""
        if conversation_history:
            history_context = "\n\nPrevious conversation context:\n"
            for i, (prev_q, prev_a) in enumerate(conversation_history[-3:], 1):  # Last 3 exchanges
                history_context += f"\nPrevious Q{i}: {prev_q}\nPrevious A{i}: {prev_a}\n"

        print("🤖 Generating answer...")
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
- Last Results are not important if Question doesn't related to it.

Current Question: {query}

Document Context:
{document_context}

Last Result:
{history_context}

Please provide your detailed answer:
"""
        answer = run_llm(prompt)

        # Store this exchange in conversation history
        conversation_history.append((query, answer))
        
        # Keep only last 5 exchanges to prevent context from getting too long
        if len(conversation_history) > 5:
            conversation_history = conversation_history[-5:]

        log_to_file(f"[{datetime.datetime.now()}] [Answer] {answer}")

        print("\n" + "="*60)
        print("📖 ANSWER:")
        print("="*60)
        print(answer)
        print("="*60)
        print(f"💾 Conversation history: {len(conversation_history)} exchanges stored")

except KeyboardInterrupt:
    print("\n👋 Goodbye!")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Please check your configuration and try again.")