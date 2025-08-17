from vector_search import search_top_k 
import subprocess
import json
import datetime

MODEL = "aya-expanse:8b"
LOG_FILE = "query_log.txt"

def run_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", MODEL, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    return result.stdout.strip()

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
        keywords = run_ollama(keyword_prompt)
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
        answer = run_ollama(prompt)

        print("\n[Answer]")
        print(answer)

except KeyboardInterrupt:
    print("\nExiting...")
