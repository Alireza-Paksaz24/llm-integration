# RAG Flask Web Application

A beautiful web interface for your RAG (Retrieval-Augmented Generation) system with multilingual support and real-time logging.

## 🚀 Features

* **Beautiful Web Interface** : Modern, responsive design with glassmorphism effects
* **Multilingual Support** : Automatic RTL/LTR text direction for Persian (فارسی), Arabic (العربية), English, Spanish, and more
* **Real-time Logging** : View system logs in real-time
* **Document Search** : Visual display of relevant documents with similarity scores
* **Conversation History** : Maintains context across multiple queries
* **Mobile Responsive** : Works perfectly on all device sizes

## 📁 Folder Structure

Create the following folder structure in your RAG directory:

```
RAG/
├── Server/
│   ├── templates/
│   │   └── index.html
│   ├── server.py
│   └── requirements.txt
├── app.py (rename to rag_app.py or keep as is)
├── embedding.py
├── vector_search.py
├── .env
├── data/
├── processed_data/
├── models/
├── embeddings.json
└── query_log.txt
```

## 🛠️ Setup Instructions

### 1. Navigate to Server Directory

```bash
cd RAG/Server
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create Templates Directory

```bash
mkdir templates
```

### 4. Place Files

* Put `server.py` in the `Server/` directory
* Put `index.html` in the `Server/templates/` directory
* Make sure your `.env` file is in the parent `RAG/` directory

### 5. Configure Environment Variables

Create or update your `.env` file in the RAG directory:

```env
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama3
LLM_API_KEY=your_api_key_if_needed
```

### 6. Prepare Your Documents

1. Place documents in the `RAG/data/` folder
2. Run the embedding script:
   ```bash
   cd ..python embedding.py
   ```

### 7. Start the Flask Server

```bash
cd Server
python server.py
```

### 8. Access the Web Interface

Open your browser and go to: `http://localhost:5000`

## 🌍 Language Support

The application automatically detects the language of your input and adjusts:

* **Text Direction** : RTL for Persian/Arabic, LTR for others
* **Font Support** : Proper rendering for all character sets
* **Language Indicator** : Shows detected language in the input field

### Supported Languages:

* 🇺🇸 English
* 🇮🇷 Persian (فارسی)
* 🇸🇦 Arabic (العربية)
* 🇪🇸 Spanish
* And many more...

## 🎛️ API Endpoints

* `GET /` - Main web interface
* `POST /api/query` - Submit queries
* `GET /api/logs` - Retrieve system logs
* `GET /api/stats` - Get document statistics
* `POST /api/clear-history` - Clear conversation history

## 🎨 Features Overview

### Chat Interface

* **Smart Input** : Auto-resizing textarea with language detection
* **Message Bubbles** : Distinct styling for user/assistant messages
* **Document References** : Shows which documents were used for answers
* **Similarity Scores** : Visual indicators of document relevance

### Logging Panel

* **Real-time Logs** : Live view of system operations
* **Categorized Entries** : Color-coded log types (queries, answers, errors)
* **Auto-scroll** : Automatically scrolls to latest entries
* **Refresh Controls** : Manual refresh and clear options

### System Statistics

* **Document Count** : Total files and chunks processed
* **Processing Info** : Average chunks per file
* **Status Indicators** : System health and availability

## 🔧 Troubleshooting

### Common Issues:

1. **Vector search not available** : Run `python embedding.py` first
2. **Import errors** : Make sure all dependencies are installed
3. **LLM connection issues** : Check your `.env` configuration
4. **Port conflicts** : Change port in `server.py` if needed

### Development Mode:

The Flask app runs in debug mode by default. For production:

```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

## 📱 Mobile Support

The interface is fully responsive and works great on:

* 📱 Mobile phones
* 📟 Tablets
* 💻 Desktop computers
* 🖥️ Large screens

## 🎯 Usage Tips

1. **Ask Clear Questions** : The system works best with specific queries
2. **Use Keywords** : Include relevant terms for better document matching
3. **Check Logs** : Monitor the logs panel for debugging
4. **Language Mixing** : You can mix languages in your queries
5. **Document Quality** : Better document preprocessing leads to better answers

## 🚀 Performance

* **Fast Response** : Optimized vector search and caching
* **Memory Efficient** : Loads embeddings once, reuses across queries
* **Scalable** : Can handle large document collections
* **Concurrent Users** : Flask supports multiple simultaneous connections

Enjoy your new RAG web interface! 🎉
