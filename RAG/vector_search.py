import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

model_name = "all-mpnet-base-v2"
if not os.path.exists(f"models/{model_name}"):
    model = SentenceTransformer(model_name)
    model.save(f"models/{model_name}")
else:
    model = SentenceTransformer(f"models/{model_name}")

with open('./embeddings.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

embeddings = np.array([item['embedding'] for item in data])

def search_top_k(query: str, k: int = 5):
    query_embedding = model.encode([query])[0].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    top_k_idx = similarities.argsort()[-k:][::-1]
    results = []
    for idx in top_k_idx:
        results.append({
            'file': data[idx]['file'],
            'chunk_index': data[idx]['chunk_index'],
            'similarity': float(similarities[idx]),
            'text': data[idx]['text']   # کل متن chunk
        })
    return results
