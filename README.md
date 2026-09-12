# 🤖 AI Enterprise Knowledge Assistant

> A production-style **Retrieval-Augmented Generation (RAG)** system for asking questions over enterprise documents with **hybrid retrieval, cross-encoder reranking, grounded generation, and source attribution**.

Built using **FastAPI + Streamlit + ChromaDB + BM25 + Sentence Transformers + Qwen 2.5 7B + Docker**.

---

## ✨ What This Project Does

The system allows users to ask natural-language questions about an enterprise document and receive:

* 🧠 Context-aware answers
* 🔎 Hybrid semantic + keyword retrieval
* 🎯 Cross-encoder reranking
* 📄 Source document and page references
* 🛡️ Grounded answers using retrieved context
* 🚫 Safe fallback when information is unavailable

### Example

**Question**

> What does Microsoft say about AI?

**System**

```text
Answer
   ↓
Relevant document context
   ↓
Source: 2025_AnnualReport.pdf
Page: 10
```

---

## 🏗️ Architecture

```text
                         👤 User
                           │
                           ▼
                  ┌─────────────────┐
                  │    Streamlit    │
                  │       UI        │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │     FastAPI     │
                  │      /ask       │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   RAG Pipeline  │
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       Dense Retrieval             BM25 Search
       SentenceTransformer         Keyword Search
              │                         │
              └────────────┬────────────┘
                           ▼
                  ┌─────────────────┐
                  │   RRF Hybrid    │
                  │    Retrieval    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Cross-Encoder   │
                  │    Reranker     │
                  └────────┬────────┘
                           │
                           ▼
                    Top 3 Chunks
                           │
                           ▼
                  ┌─────────────────┐
                  │   Qwen 2.5 7B   │
                  │     Ollama      │
                  └────────┬────────┘
                           │
                           ▼
                Answer + Source Pages
```

---

## 🔄 RAG Pipeline

```text
PDF
 │
 ▼
Text Extraction
 │
 ▼
Chunking
 │
 ▼
Embeddings
 │
 ▼
ChromaDB
 │
 ├───────────────┐
 ▼               ▼
Dense Search    BM25 Search
 │               │
 └───────┬───────┘
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

## 🔎 Retrieval System

### 1. Dense Retrieval

Queries are converted into embeddings using:

```text
all-MiniLM-L6-v2
```

The embeddings are searched against document vectors stored in **ChromaDB**.

This captures semantic similarity even when the query does not use exactly the same words as the document.

---

### 2. BM25 Retrieval

BM25 provides keyword-based retrieval.

It is particularly useful for:

* Names
* Numbers
* Exact terminology
* Important keywords

---

### 3. Hybrid Retrieval

Dense and BM25 rankings are combined using:

```text
Reciprocal Rank Fusion (RRF)
```

This combines semantic and lexical retrieval signals before reranking.

---

### 4. Cross-Encoder Reranking

Hybrid candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The cross-encoder evaluates each:

```text
Query + Document Chunk
```

pair and produces a refined relevance ranking.

---

## 🤖 Grounded Generation

The final answer is generated using:

```text
Qwen 2.5 7B
       +
Ollama
```

The model receives only the retrieved context and is instructed not to invent information.

When the required information is unavailable, the system returns:

```text
I don't have enough information in the provided documents.
```

This provides a simple guard against unsupported answers.

---

## 📚 Source Attribution

Each document chunk stores metadata such as:

```json
{
  "source": "2025_AnnualReport.pdf",
  "page": 10
}
```

The API returns the answer together with the relevant source pages.

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

# 📊 Retrieval Evaluation

The retrieval pipeline was evaluated using a manually prepared question set.

### Metrics

* Hit@3
* Hit@5
* Recall@10
* MRR@10
* Average latency

### Results

| Metric          |    Dense |     Hybrid |   Reranker |
| --------------- | -------: | ---------: | ---------: |
| **Hit@3**       |   31.03% |     37.93% | **46.55%** |
| **Hit@5**       |   39.66% |     44.83% | **48.28%** |
| **Recall@10**   |   44.83% | **56.90%** | **56.90%** |
| **MRR@10**      |   0.2685 |     0.2893 | **0.3840** |
| **Avg Latency** | 89.29 ms |   90.33 ms |  311.59 ms |

### Retrieval progression

```text
Dense
  │
  ├── Recall@10: 44.83%
  │
  ▼
Hybrid
  │
  ├── Recall@10: 56.90%
  │
  ▼
Reranker
  │
  ├── Recall@10: 56.90%
  └── MRR@10:    0.3840
```

The evaluation also identified **19 queries that remained unresolved after reranking**, providing clear areas for future retrieval improvements.

---

## 🧪 Failure Analysis

The evaluation tracked retrieval failures across each stage.

```text
Dense failures:      7
Hybrid failures:     4
Reranker failures:  5
```

Hybrid retrieval recovered:

```text
3 dense failures
```

The evaluation showed that retrieval quality is not determined by reranking alone and that document-aware chunking, query processing, and retrieval configuration remain important areas for improvement.

---

# 🛠️ Tech Stack

| Technology                | Role              |
| ------------------------- | ----------------- |
| **Python**                | Core development  |
| **FastAPI**               | Backend API       |
| **Streamlit**             | Web interface     |
| **ChromaDB**              | Vector database   |
| **Sentence Transformers** | Embeddings        |
| **BM25**                  | Sparse retrieval  |
| **RRF**                   | Hybrid ranking    |
| **Cross-Encoder**         | Reranking         |
| **Ollama**                | Local LLM runtime |
| **Qwen 2.5 7B**           | LLM               |
| **LangChain**             | LLM integration   |
| **Docker**                | Containerization  |

---

# 📁 Project Structure

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
└── chroma_db/
    └── ...
```

> `venv/` is used only for local development and should not be committed to GitHub.

---

# ⚙️ Local Setup

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>

cd ai-enterprise-knowledge-assistant
```

## 2. Create virtual environment

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🦙 Ollama Setup

Install Ollama and pull the required model:

```bash
ollama pull qwen2.5:7b
```

Verify:

```bash
ollama list
```

---

# ▶️ Run FastAPI

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

# 🖥️ Run Streamlit

In another terminal:

```bash
streamlit run streamlit_app.py
```

The Streamlit interface can then be used to ask questions about the enterprise document.

---

# 🐳 Docker

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

When FastAPI runs inside Docker while Ollama runs on the host machine, the Ollama endpoint can use:

```text
http://host.docker.internal:11434
```

---

# 💬 Example Questions

```text
What does Microsoft say about AI?

What are Microsoft's major business segments?

What does Microsoft say about cloud computing?

What are Microsoft's AI offerings?
```

Questions outside the available document context are handled using the system's grounded-answer fallback.

---

# 🎯 Engineering Concepts Demonstrated

This project demonstrates practical implementation of:

```text
Document Ingestion
       ↓
Chunking
       ↓
Embeddings
       ↓
Vector Search
       +
Keyword Search
       ↓
Hybrid Retrieval
       ↓
RRF
       ↓
Cross-Encoder Reranking
       ↓
Context Selection
       ↓
Grounded LLM Generation
       ↓
Source Attribution
```

It also includes:

* Retrieval evaluation
* Failure analysis
* API development
* Local LLM inference
* Docker containerization
* Source-aware responses

---

# 🔮 Future Improvements

Potential improvements include:

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

# 👨‍💻 About

Built as a **GenAI / AI Engineering portfolio project** to demonstrate practical experience with:

**RAG • Retrieval Systems • LLMs • APIs • Evaluation • Docker**

---
