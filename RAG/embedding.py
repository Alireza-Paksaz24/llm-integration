import os
import json
import csv
from sentence_transformers import SentenceTransformer
import PyPDF2

model_name = "all-mpnet-base-v2"
if not os.path.exists(f"models/{model_name}"):
    model = SentenceTransformer(model_name)
    model.save(f"models/{model_name}")
else:
    model = SentenceTransformer(f"models/{model_name}")



# تابع برای تقسیم متن به تکه‌های 500 کلمه‌ای (یا دلخواه)
def chunk_text(text, max_words=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_md(file_path):
    return read_txt(file_path)

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = []
    def extract_text(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                extract_text(v)
        elif isinstance(obj, list):
            for item in obj:
                extract_text(item)
        elif isinstance(obj, str):
            texts.append(obj)
    extract_text(data)
    return "\n".join(texts)

def read_csv(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            texts.append(" ".join(row))
    return "\n".join(texts)

def read_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text += txt + "\n"
    return text

def get_text_from_file(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'txt':
        return read_txt(file_path)
    elif ext == 'md':
        return read_md(file_path)
    elif ext == 'json':
        return read_json(file_path)
    elif ext == 'csv':
        return read_csv(file_path)
    elif ext == 'pdf':
        return read_pdf(file_path)
    else:
        print(f"Format {ext} not supported: {file_path}")
        return ""

def process_folder(folder_path, output_file='embeddings.json'):
    all_embeddings = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)
            print(f"Reading file: {path}")
            text = get_text_from_file(path)
            if not text.strip():
                continue

            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                embedding = model.encode(chunk)
                all_embeddings.append({
                    'file': path,
                    'chunk_index': i,
                    'embedding': embedding.tolist(),
                    'text': chunk
                })

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(all_embeddings, f_out, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_embeddings)} embeddings in {output_file}")

if __name__ == '__main__':
    folder = "./data"
    process_folder(folder)
