import os
import json
import csv
import shutil
import hashlib
from datetime import datetime
from sentence_transformers import SentenceTransformer
import PyPDF2

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("processed_data", exist_ok=True)

# Load or download model
model_name = "all-mpnet-base-v2"
model_path = f"models/{model_name}"

print(f"Checking for model at: {model_path}")

if not os.path.exists(model_path):
    print(f"📥 Downloading model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print(f"💾 Saving model to '{model_path}'...")
    model.save(model_path)
    print("✅ Model downloaded and cached!")
else:
    print(f"✅ Loading cached model from '{model_path}'...")
    model = SentenceTransformer(model_path)
    print("✅ Model loaded successfully!")

def get_file_hash(file_path):
    """Generate MD5 hash of file content for change detection"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"❌ Error hashing {file_path}: {e}")
        return None

def load_existing_embeddings(embeddings_file='embeddings.json'):
    """Load existing embeddings and file tracking info"""
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both old and new format
            if isinstance(data, dict) and 'embeddings' in data:
                # New format with metadata
                embeddings = data['embeddings']
                processed_files = data.get('processed_files', {})
            else:
                # Old format - just embeddings list
                embeddings = data
                processed_files = {}
                # Build processed_files from existing embeddings
                for item in embeddings:
                    file_path = item['file']
                    if file_path not in processed_files:
                        processed_files[file_path] = {
                            'hash': None,  # Unknown for old files
                            'processed_date': 'unknown',
                            'chunks': 0
                        }
                    processed_files[file_path]['chunks'] += 1
            
            print(f"✅ Loaded {len(embeddings)} existing embeddings from {len(processed_files)} files")
            return embeddings, processed_files
        except Exception as e:
            print(f"❌ Error loading existing embeddings: {e}")
            return [], {}
    else:
        print("📝 No existing embeddings found, starting fresh")
        return [], {}

def save_embeddings(embeddings, processed_files, embeddings_file='embeddings.json'):
    """Save embeddings with metadata"""
    data = {
        'metadata': {
            'total_embeddings': len(embeddings),
            'total_files': len(processed_files),
            'last_updated': datetime.now().isoformat(),
            'model_used': model_name
        },
        'processed_files': processed_files,
        'embeddings': embeddings
    }
    
    with open(embeddings_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    with open(f'Server/{embeddings_file}', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def chunk_text(text, max_words=500):
    """Split text into chunks of max_words"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def read_txt(file_path):
    """Read plain text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()

def read_md(file_path):
    """Read markdown file"""
    return read_txt(file_path)

def read_json(file_path):
    """Read JSON file and extract all text content"""
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
            if obj.strip():
                texts.append(obj)
    
    extract_text(data)
    return "\n".join(texts)

def read_csv(file_path):
    """Read CSV file"""
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            for row in reader:
                if row:
                    texts.append(" ".join(str(cell) for cell in row))
    except Exception as e:
        print(f"Error reading CSV {file_path}: {e}")
    return "\n".join(texts)

def read_pdf(file_path):
    """Read PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    print(f"Error reading page {page_num} from {file_path}: {e}")
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def get_text_from_file(file_path):
    """Extract text from various file formats"""
    ext = file_path.split('.')[-1].lower()
    
    try:
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
            print(f"⚠️  Unsupported format '{ext}' for file: {file_path}")
            return ""
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return ""

def move_processed_file(source_path, processed_folder="processed_data"):
    """Move processed file to processed_data folder maintaining folder structure"""
    try:
        # Get relative path from data folder
        rel_path = os.path.relpath(source_path, "data")
        dest_path = os.path.join(processed_folder, rel_path)
        
        # Create destination directory if needed
        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Move the file
        shutil.move(source_path, dest_path)
        print(f"📁 Moved: {source_path} → {dest_path}")
        return dest_path
    except Exception as e:
        print(f"❌ Error moving {source_path}: {e}")
        return source_path

def find_new_and_changed_files(data_folder, processed_files):
    """Find files that are new or have changed"""
    new_files = []
    changed_files = []
    
    for root, dirs, files in os.walk(data_folder):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.startswith('.') or file.endswith(('.pyc', '.DS_Store')):
                continue
                
            file_path = os.path.join(root, file)
            current_hash = get_file_hash(file_path)
            
            if current_hash is None:
                continue
            
            if file_path not in processed_files:
                # New file
                new_files.append(file_path)
            else:
                # Check if file has changed
                stored_hash = processed_files[file_path].get('hash')
                if stored_hash != current_hash:
                    changed_files.append(file_path)
    
    return new_files, changed_files

def process_incremental(data_folder="./data", embeddings_file='embeddings.json', move_files=True):
    """Process only new and changed files incrementally"""
    
    if not os.path.exists(data_folder):
        print(f"❌ Data folder not found: {data_folder}")
        return
    
    print(f"🔍 Checking for new and changed files in: {data_folder}")
    
    # Load existing embeddings
    all_embeddings, processed_files = load_existing_embeddings(embeddings_file)
    
    # Find new and changed files
    new_files, changed_files = find_new_and_changed_files(data_folder, processed_files)
    
    if not new_files and not changed_files:
        print("✅ No new or changed files found. Everything is up to date!")
        return
    
    print(f"📁 Found {len(new_files)} new files and {len(changed_files)} changed files")
    
    files_to_process = new_files + changed_files
    new_embeddings = []
    
    # Remove embeddings for changed files
    if changed_files:
        print("🗑️  Removing old embeddings for changed files...")
        all_embeddings = [
            emb for emb in all_embeddings 
            if emb['file'] not in changed_files
        ]
    
    # Process files
    for file_path in files_to_process:
        status = "NEW" if file_path in new_files else "CHANGED"
        print(f"📖 Processing {status}: {file_path}")
        
        text = get_text_from_file(file_path)
        if not text.strip():
            print(f"⚠️  No text extracted from: {file_path}")
            continue
        
        chunks = chunk_text(text)
        print(f"   📝 Created {len(chunks)} chunks")
        
        file_embeddings = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                try:
                    embedding = model.encode(chunk)
                    embedding_data = {
                        'file': file_path,
                        'chunk_index': i,
                        'embedding': embedding.tolist(),
                        'text': chunk
                    }
                    file_embeddings.append(embedding_data)
                except Exception as e:
                    print(f"❌ Error creating embedding for chunk {i} in {file_path}: {e}")
        
        new_embeddings.extend(file_embeddings)
        
        # Update processed files tracking
        processed_files[file_path] = {
            'hash': get_file_hash(file_path),
            'processed_date': datetime.now().isoformat(),
            'chunks': len(file_embeddings)
        }
        
        # Move file to processed folder if requested
        if move_files:
            new_file_path = move_processed_file(file_path)
            # Update file paths in embeddings and processed_files
            for emb in file_embeddings:
                emb['file'] = new_file_path
            processed_files[new_file_path] = processed_files.pop(file_path)
    
    # Combine all embeddings
    all_embeddings.extend(new_embeddings)
    
    print(f"\n📊 Processing Summary:")
    print(f"   New files processed: {len(new_files)}")
    print(f"   Changed files processed: {len(changed_files)}")
    print(f"   New embeddings created: {len(new_embeddings)}")
    print(f"   Total embeddings: {len(all_embeddings)}")
    
    # Save updated embeddings
    if all_embeddings:
        print(f"💾 Saving embeddings to {embeddings_file}...")
        save_embeddings(all_embeddings, processed_files, embeddings_file)
        print("✅ Embeddings saved successfully!")
    
    return len(new_files), len(changed_files), len(new_embeddings)

def show_status():
    """Show current status of embeddings and files"""
    print("📊 Current Status:")
    
    # Check data folder
    data_files = []
    if os.path.exists("data"):
        for root, dirs, files in os.walk("data"):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if not file.startswith('.') and not file.endswith(('.pyc', '.DS_Store')):
                    data_files.append(os.path.join(root, file))
    
    print(f"   📁 Files in data folder: {len(data_files)}")
    
    # Check processed files
    processed_files = []
    if os.path.exists("processed_data"):
        for root, dirs, files in os.walk("processed_data"):
            for file in files:
                if not file.startswith('.'):
                    processed_files.append(os.path.join(root, file))
    
    print(f"   📁 Files in processed_data folder: {len(processed_files)}")
    
    # Check embeddings
    if os.path.exists("embeddings.json"):
        embeddings, _ = load_existing_embeddings("embeddings.json")
        print(f"   📝 Total embeddings: {len(embeddings)}")
    else:
        print("   📝 No embeddings file found")
    
    if data_files:
        print(f"\n📋 Files ready to process:")
        for f in data_files[:5]:  # Show first 5
            print(f"   - {f}")
        if len(data_files) > 5:
            print(f"   ... and {len(data_files) - 5} more")

if __name__ == '__main__':
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "status":
            show_status()
            exit(0)
        elif command == "no-move":
            print("🔄 Processing files without moving them...")
            process_incremental(move_files=False)
            exit(0)
    
    # Default behavior
    show_status()
    
    # Ask user if they want to proceed
    if os.listdir("data") if os.path.exists("data") else []:
        proceed = input("\nProceed with processing? (y/N): ").strip().lower()
        if proceed in ['y', 'yes']:
            process_incremental()
        else:
            print("👋 Processing cancelled.")
    else:
        print("\n💡 Add some files to the 'data' folder and run again!")