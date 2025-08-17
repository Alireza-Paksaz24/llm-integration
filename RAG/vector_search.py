import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load or download model (same as embedding.py for consistency)
model_name = "all-mpnet-base-v2"
model_path = f"models/{model_name}"

if not os.path.exists(model_path):
    model_path =f"../models/{model_name}"

if not os.path.exists(model_path):
    print(f"📥 Downloading model '{model_name}' for search...")
    model = SentenceTransformer(model_name)
    print(f"💾 Saving model to '{model_path}'...")
    model.save(model_path)
    print("✅ Model downloaded and cached!")
else:
    model = SentenceTransformer(model_path)

# Load embeddings
embeddings_file = './embeddings.json'
if not os.path.exists(embeddings_file):
    raise FileNotFoundError(f"""
❌ Embeddings file not found: {embeddings_file}

Please run the embedding script first:
    python embedding.py

This will process your documents and create the embeddings.json file.
""")

# Load embeddings
embeddings_file = './embeddings.json'
if not os.path.exists(embeddings_file):
    raise FileNotFoundError(f"""
❌ Embeddings file not found: {embeddings_file}

Please run the embedding script first:
    python embedding.py

This will process your documents and create the embeddings.json file.
""")

print(f"📚 Loading embeddings from {embeddings_file}...")
with open(embeddings_file, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

# Handle both old and new format
if isinstance(full_data, dict) and 'embeddings' in full_data:
    # New format with metadata
    data = full_data['embeddings']
    metadata = full_data.get('metadata', {})
    processed_files = full_data.get('processed_files', {})
    print(f"📊 Metadata: {metadata.get('total_files', 'unknown')} files, last updated: {metadata.get('last_updated', 'unknown')}")
else:
    # Old format - just embeddings list
    data = full_data
    processed_files = {}

print(f"✅ Loaded {len(data)} document chunks")

# Convert embeddings to numpy array for faster computation
embeddings = np.array([item['embedding'] for item in data])
print(f"📊 Embeddings shape: {embeddings.shape}")

def search_top_k(query: str, k: int = 5):
    """
    Search for top-k most similar document chunks
    
    Args:
        query (str): Search query
        k (int): Number of top results to return
    
    Returns:
        list: List of dictionaries with file, chunk_index, similarity, text
    """
    if not query.strip():
        print("⚠️  Empty query provided")
        return []
    
    try:
        # Encode the query
        query_embedding = model.encode([query])[0].reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Get top-k indices (sorted in descending order)
        top_k_idx = similarities.argsort()[-k:][::-1]
        
        # Build results
        results = []
        for idx in top_k_idx:
            result = {
                'file': data[idx]['file'],
                'chunk_index': data[idx]['chunk_index'],
                'similarity': float(similarities[idx]),
                'text': data[idx]['text']
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return []

def search_with_threshold(query: str, threshold: float = 0.3, max_results: int = 10):
    """
    Search for document chunks above similarity threshold
    
    Args:
        query (str): Search query
        threshold (float): Minimum similarity score (0-1)
        max_results (int): Maximum number of results
    
    Returns:
        list: Filtered results above threshold
    """
    if not query.strip():
        return []
    
    try:
        query_embedding = model.encode([query])[0].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Filter by threshold and get top results
        valid_indices = np.where(similarities >= threshold)[0]
        if len(valid_indices) == 0:
            print(f"⚠️  No results found above threshold {threshold}")
            return []
        
        # Sort by similarity and take top results
        sorted_indices = valid_indices[similarities[valid_indices].argsort()[::-1]]
        top_indices = sorted_indices[:max_results]
        
        results = []
        for idx in top_indices:
            results.append({
                'file': data[idx]['file'],
                'chunk_index': data[idx]['chunk_index'], 
                'similarity': float(similarities[idx]),
                'text': data[idx]['text']
            })
        
        return results
        
    except Exception as e:
        print(f"❌ Error during threshold search: {e}")
        return []

def get_document_stats():
    """Get statistics about the loaded documents"""
    if not data:
        return {}
    
    # Count files and chunks
    files = set(item['file'] for item in data)
    
    # Count chunks per file
    file_chunks = {}
    for item in data:
        file_name = item['file']
        if file_name in file_chunks:
            file_chunks[file_name] += 1
        else:
            file_chunks[file_name] = 1
    
    return {
        'total_chunks': len(data),
        'total_files': len(files),
        'files': list(files),
        'chunks_per_file': file_chunks,
        'avg_chunks_per_file': len(data) / len(files) if files else 0
    }

# Test function
def test_search(query="test query", k=3):
    """Test the search functionality"""
    print(f"\n🔍 Testing search with query: '{query}'")
    results = search_top_k(query, k)
    
    if results:
        print(f"📋 Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. File: {result['file']}")
            print(f"   Chunk: {result['chunk_index']}")
            print(f"   Similarity: {result['similarity']:.4f}")
            print(f"   Text preview: {result['text'][:100]}...")
    else:
        print("❌ No results found")
    
    return results

if __name__ == '__main__':
    # Show document statistics
    stats = get_document_stats()
    print(f"\n📊 Document Statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Average chunks per file: {stats['avg_chunks_per_file']:.1f}")
    
    # Test search
    test_query = input("\nEnter a test query (or press Enter to skip): ").strip()
    if test_query:
        test_search(test_query)