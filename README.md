````markdown
# AI Enterprise Knowledge Assistant

A production-style Retrieval-Augmented Generation (RAG) system that allows users to ask questions about enterprise documents and receive grounded answers with source references.

The system uses hybrid retrieval, reranking, and a local LLM to improve answer relevance and reduce hallucination.

---

## 🚀 Features

- PDF document ingestion
- Recursive text chunking
- Sentence Transformer embeddings
- ChromaDB vector database
- Dense vector search
- BM25 sparse keyword search
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Cross-Encoder reranking
- Grounded LLM generation
- Source/page references
- FastAPI backend
- Streamlit user interface
- Docker containerization
- Retrieval evaluation using:
  - Hit@3
  - Hit@5
  - Recall@10
  - MRR@10
  - Latency

---

## 🏗️ Architecture

```text
                         User
                           │
                           ▼
                   ┌───────────────┐
                   │   Streamlit   │
                   │      UI       │
                   └───────┬───────┘
                           │
                           ▼
                   ┌───────────────┐
                   │    FastAPI    │
                   │     /ask      │
                   └───────┬───────┘
                           │
                           ▼
                 ┌─────────────────────┐
                 │     RAG Pipeline     │
                 │                     │
                 │ Query Embedding     │
                 │        ↓            │
                 │ Dense Search        │
                 │        +            │
                 │ BM25 Search         │
                 │        ↓            │
                 │ Hybrid RRF          │
                 │        ↓            │
                 │ Cross-Encoder       │
                 │ Reranking           │
                 │        ↓            │
                 │ Top Context         │
                 │        ↓            │
                 │ Qwen 2.5 7B         │
                 └──────────┬──────────┘
                            │
                            ▼
                    Answer + Sources
````

---

## 🔄 RAG Pipeline

The system follows this pipeline:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
ChromaDB
 ↓
       ┌───────────────┐
       │ Dense Search  │
       └───────┬───────┘
               │
               ├──────────────┐
               │              │
               ▼              ▼
        Vector Search      BM25 Search
               │              │
               └──────┬───────┘
                      ▼
                 RRF Hybrid
                      │
                      ▼
              Cross-Encoder
                 Reranking
                      │
                      ▼
                Top 3 Chunks
                      │
                      ▼
                Qwen 2.5 7B
                      │
                      ▼
              Grounded Answer
                      │
                      ▼
              Source + Page
```

---

## 🧠 Retrieval Strategy

### 1. Dense Retrieval

The user query is converted into an embedding using:

```text
all-MiniLM-L6-v2
```

The embedding is compared against document chunk embeddings stored in ChromaDB.

This helps retrieve semantically similar content even when the exact words are different.

---

### 2. BM25 Retrieval

BM25 provides keyword-based retrieval.

This is useful when queries contain important exact terms, names, numbers, or terminology.

---

### 3. Hybrid Retrieval

Dense retrieval and BM25 results are combined using:

```text
Reciprocal Rank Fusion (RRF)
```

This combines the rankings from both retrieval methods.

---

### 4. Cross-Encoder Reranking

The top hybrid candidates are passed through:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The cross-encoder evaluates the relevance of each query-document pair and produces a refined ranking.

---

## 🤖 LLM

The final answer is generated using:

```text
Qwen 2.5 7B
```

through Ollama.

The LLM is instructed to use only the retrieved context.

If the required information is not present, the system returns:

```text
I don't have enough information in the provided documents.
```

This helps reduce unsupported answers and hallucinations.

---

## 📚 Source Attribution

The system keeps metadata for every document chunk.

Example:

```json
{
  "source": "2025_AnnualReport.pdf",
  "page": 10
}
```

The final API response includes the relevant source pages.

Example:

```json
{
  "answer": "Microsoft says that AI is fundamentally transforming productivity...",
  "sources": [
    {
      "source": "2025_AnnualReport.pdf",
      "page": 10
    }
  ]
}
```

---

## 📊 Retrieval Evaluation

The retrieval system was evaluated using a manually prepared question set.

Metrics include:

* Hit@3
* Hit@5
* Recall@10
* MRR@10
* Average latency

The evaluation compares:

```text
Dense Retrieval
       ↓
Hybrid Retrieval
       ↓
Reranked Retrieval
```

Example evaluation output:

```text
Dense
Hit@3:      31.03%
Hit@5:      39.66%
Recall@10:  44.83%
MRR@10:     0.2685

Hybrid
Hit@3:      37.93%
Hit@5:      44.83%
Recall@10:  56.90%
MRR@10:     0.2893

Reranker
Hit@3:      46.55%
Hit@5:      48.28%
Recall@10:  56.90%
MRR@10:     0.3840
```

---

## 🛠️ Tech Stack

| Technology            | Purpose           |
| --------------------- | ----------------- |
| Python                | Core development  |
| FastAPI               | Backend API       |
| Streamlit             | User interface    |
| ChromaDB              | Vector database   |
| Sentence Transformers | Embeddings        |
| BM25                  | Sparse retrieval  |
| Cross-Encoder         | Reranking         |
| Ollama                | Local LLM runtime |
| Qwen 2.5 7B           | LLM               |
| LangChain             | LLM integration   |
| Docker                | Containerization  |

---

## 📁 Project Structure

```text
ai-enterprise-knowledge-assistant/
│
├── app.py
├── rag_pipeline.py
├── streamlit_app.py
├── evaluate_retrieval.py
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
│
├── data/
│   └── 2025_AnnualReport.pdf
│
├── chroma_db/
│   └── ...
│
└── venv/
    └── ...
```

`venv/` is used locally and should not be committed to GitHub.

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ai-enterprise-knowledge-assistant
```

### 2. Create virtual environment

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Setup Ollama

Install Ollama and make sure the Qwen model is available:

```bash
ollama pull qwen2.5:7b
```

Check:

```bash
ollama list
```

---

## ▶️ Run FastAPI

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## 🖥️ Run Streamlit

```bash
streamlit run streamlit_app.py
```

The Streamlit interface can then be used to ask questions about the document.

---

## 🐳 Run with Docker

Build the image:

```bash
docker build -t ai-enterprise-knowledge-assistant .
```

Run:

```bash
docker run --rm -p 8000:8000 ai-enterprise-knowledge-assistant
```

Open:

```text
http://localhost:8000/docs
```

### Docker + Ollama

When FastAPI runs inside Docker and Ollama runs on the Mac host, the Ollama endpoint is configured using:

```text
http://host.docker.internal:11434
```

---

## 🧪 Example Questions

```text
What does Microsoft say about AI?

What are Microsoft's major business segments?

What does Microsoft say about cloud computing?

What are Microsoft's AI offerings?
```

For questions outside the available document context, the system is designed to avoid inventing an answer.

---

## 🎯 Project Objective

The objective of this project is to demonstrate how a production-style enterprise knowledge assistant can combine:

```text
Semantic Retrieval
+
Keyword Retrieval
+
Hybrid Ranking
+
Reranking
+
Grounded Generation
```

to answer questions from enterprise documents.

---

## 🔮 Future Improvements

Possible future improvements include:

* Better document-aware chunking
* Query rewriting
* Metadata-aware retrieval
* Improved BM25 tokenization
* Hybrid score tuning
* Reranker threshold optimization
* Streaming LLM responses
* Conversation memory
* Authentication
* Multi-document support
* Observability and tracing
* Cloud deployment

---

## 👨‍💻 Author

Built as an AI/GenAI engineering portfolio project demonstrating practical RAG, retrieval evaluation, API development, and containerization.

```
```
