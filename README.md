# 🤖 Advanced RAG Chatbot (Evidence-Based Management Assistant)

An advanced **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit, LangChain, FAISS, and Groq LLM**, designed to provide **evidence-based management insights** directly from PDF documents.

This system follows a **strict context-grounded approach**, ensuring responses are based only on retrieved data — no hallucinations, no assumptions.

---

## 🚀 Features

- 🔍 **Hybrid Retrieval System**
  - Dense retrieval (FAISS embeddings)
  - Sparse retrieval (BM25 ranking)
  
- ⚡ **Query Expansion (HyDE-style)**
  - Automatically generates multiple search queries for better retrieval

- 🧠 **Parallel Retrieval**
  - Multi-threaded document fetching for faster responses

- 📊 **Relevance Filtering**
  - LLM-based grading (YES/NO) to filter most relevant chunks

- 🧩 **Knowledge Graph Integration**
  - Extracts entity relationships from documents
  - Enhances semantic matching

- 🧾 **Source Transparency**
  - Shows exact PDF snippets with page numbers

- ✨ **Optional Summarization**
  - Clean summaries of responses

- 💬 **Chat Memory (Session-based)**

---

## 🏗️ Tech Stack

- **Frontend**: Streamlit  
- **LLM**: Groq (LLaMA 3.1 8B Instant)  
- **Embeddings**: SentenceTransformers  
- **Vector DB**: FAISS  
- **Retrieval**: BM25 + Dense Hybrid  
- **PDF Processing**: PyMuPDF  
- **Orchestration**: LangChain  

---

## 📁 Project Structure

project-root/
│
├── data/
│ └── pdf/ # Your PDF documents
│
├── faiss_db/ # Stored FAISS index
│
├── app/
│ └── main.py # Main Streamlit app
│
├── .env # API keys
├── requirements.txt
└── README.md


---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Create Virtual Environment

python -m venv venv

source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

3. Install Dependencies

pip install -r requirements.txt
🔑 Environment Variables

Create a .env file in the root directory:

GROQ_API_KEY=your_api_key_here

▶️ Run the App

streamlit run app/main.py

🧠 How It Works

1. Document Processing

PDFs are loaded and split into chunks
Each chunk is enriched with Knowledge Graph triples

2. Embedding & Storage

Text is converted into vector embeddings
Stored in FAISS vector database

3. Query Handling

User query is expanded into multiple variations
Retrieval happens using:
Dense similarity (FAISS)
Sparse ranking (BM25)

4. Relevance Filtering

Top chunks are evaluated using LLM (YES/NO grading)

5. Response Generation

Only verified context is passed to the LLM
Response is generated with strict grounding

🎯 Use Cases
📊 Management Consulting
💼 Business Decision Support
📚 Research Assistant
📖 Document Q&A Systems
🧾 Financial Analysis from Reports
🛡️ Key Design Principles
❌ No hallucination
✅ Context-only answers
🔍 Transparent sources
⚖️ Balanced evidence-based advice
📸 UI Features

Chat interface (like ChatGPT)
Expandable:

✨ Summary
🔍 Source verification

Adjustable chunk retrieval

⚡ Performance Optimizations

Parallel retrieval using ThreadPoolExecutor
Reduced query expansion overhead
Cached FAISS loading (@st.cache_resource)
Retry + rate-limit handling

🔮 Future Improvements
🔁 Persistent chat memory (database)
🌐 API deployment (FastAPI)
📊 Analytics dashboard
🧠 Better reranking models
📦 Docker support
🤝 Contributing

Pull requests are welcome!
If you'd like to improve performance, UI, or add features — feel free to contribute.

## 📜 License

This project is for portfolio and demonstration purposes only.  
Commercial use, redistribution, or reproduction is not allowed without permission.

👨‍💻 Author

Your Name

GitHub: https://github.com/
LinkedIn: https://linkedin.com/in/your-profile
⭐ Support

If you like this project, don’t forget to star the repo ⭐