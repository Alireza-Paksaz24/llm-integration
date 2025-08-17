import os
import json
import datetime
import requests
from dotenv import load_dotenv
from vector_search import search_top_k

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
    import subprocess
    result = subprocess.run(
        ["ollama", "run", LLM_MODEL, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    return result.stdout.strip()


def run_lmstudio(prompt: str) -> str:
    resp = requests.post(
        f"{LLM_BASE_URL}/v1/chat/completions",
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


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


try:
    while True:
        query = input("\nEnter your query (or Ctrl+C to exit): ").strip()
        if not query:
            continue

        keyword_prompt = f"""
Extract only the main keywords or key phrases from the following user query for a vector search.
Return keywords separated by commas, no explanation.

Query: {query}
"""
        keywords = run_llm(keyword_prompt)
        log_to_file(f"[{datetime.datetime.now()}] [Search Keywords] {keywords}")

        top_k_docs = search_top_k(keywords)
        log_to_file(f"[{datetime.datetime.now()}] [Search Results]:")
        for doc in top_k_docs:
            log_to_file(f"- File: {doc['file']} (Chunk {doc['chunk_index']})")
            log_to_file(f"  Similarity: {doc['similarity']:.4f}")
            log_to_file(f"  Text: {doc['text']}\n")

        context = "\n\n".join([doc['text'] for doc in top_k_docs])

        prompt = f"""
You are a knowledgeable assistant answering the user's question using only the information provided below.

Instructions:
- Use the context as your sole source.
- If the answer is fully in the context, explain clearly in your own words.
- If partially present, combine logically.
- If insufficient info, say so politely.
- Avoid verbatim copying unless needed.
- Be concise and friendly.

Context:
{context}

Question:
{query}

Please provide your detailed answer:
"""
        answer = run_llm(prompt)

        print("\n[Answer]")
        print(answer)

except KeyboardInterrupt:
    print("\nExiting...")
