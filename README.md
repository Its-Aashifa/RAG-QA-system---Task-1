# RAG Question Answering API

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, FAISS, and Groq LLM. Upload documents (PDF or TXT), then ask natural language questions answered by the content of those documents.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                        CLIENT                            │
│              (curl / Postman / Frontend)                 │
└─────────────────────┬────────────────────────────────────┘
                      │ HTTP
┌─────────────────────▼────────────────────────────────────┐
│                   FastAPI App                            │
│   ┌────────────┐  ┌─────────────┐  ┌────────────────┐   │
│   │  /upload   │  │/jobs/{id}   │  │    /query      │   │
│   └─────┬──────┘  └──────┬──────┘  └───────┬────────┘   │
│         │                │                  │            │
│   ┌─────▼──────┐  ┌──────▼──────┐  ┌───────▼────────┐   │
│   │ Job Manager│  │  Job Store  │  │ Vector Store   │   │
│   │(ThreadPool)│  │ (in-memory) │  │    (FAISS)     │   │
│   └─────┬──────┘  └─────────────┘  └───────┬────────┘   │
│         │                                   │            │
│   ┌─────▼───────────────────────────────────▼────────┐   │
│   │          Document Ingestion Pipeline              │   │
│   │  Parser → Chunker → Embedder → FAISS Index       │   │
│   └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │      Groq API         │
                    │   (LLaMA 3 70B)       │
                    └───────────────────────┘
```

See `docs/design_decisions.md` for architecture rationale, chunk size justification, observed retrieval failure cases, and latency metrics.

---

## Features

- ✅ PDF and TXT document ingestion
- ✅ Sentence-aware sliding window chunking (512 chars, 64 overlap)
- ✅ Local embeddings via `sentence-transformers/all-MiniLM-L6-v2`
- ✅ FAISS IndexFlatIP for cosine similarity search
- ✅ Background ingestion jobs (ThreadPoolExecutor)
- ✅ LLM answer generation via Groq (LLaMA 3 70B)
- ✅ Pydantic request/response validation
- ✅ Per-IP rate limiting (SlowAPI)
- ✅ Latency tracking on every query response

---

## Setup

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier works)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/rag-qa-api.git
cd rag-qa-api
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here
```

### 3. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: `http://localhost:8000/docs`

---

## Usage

### Upload a document

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@your_document.pdf"
```

Response:
```json
{
  "job_id": "abc123",
  "filename": "your_document.pdf",
  "status": "pending",
  "message": "Document accepted. Ingestion started in background."
}
```

### Poll ingestion status

```bash
curl http://localhost:8000/api/v1/jobs/abc123
```

Response when complete:
```json
{
  "job_id": "abc123",
  "filename": "your_document.pdf",
  "status": "completed",
  "chunks_created": 47
}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings of the report?", "top_k": 5}'
```

Response:
```json
{
  "question": "What are the main findings of the report?",
  "answer": "According to the document, the main findings are...",
  "retrieved_chunks": [
    {
      "text": "The study found that...",
      "source": "your_document.pdf",
      "chunk_index": 12,
      "similarity_score": 0.87
    }
  ],
  "latency_ms": 342.5,
  "model_used": "llama3-70b-8192"
}
```

### Query a specific document

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "document_id": "the-document-id-from-upload"}'
```

---

## Project Structure

```
rag-qa-api/
├── main.py                    # FastAPI app entry point
├── requirements.txt
├── .env.example
├── app/
│   ├── api/
│   │   └── routes.py          # API endpoints + rate limiting
│   ├── core/
│   │   └── config.py          # Pydantic Settings
│   ├── models/
│   │   └── schemas.py         # Request/response Pydantic models
│   └── services/
│       ├── parser.py          # PDF + TXT document parsing
│       ├── chunker.py         # Sentence-aware sliding window chunking
│       ├── vector_store.py    # FAISS embedding + retrieval
│       ├── llm.py             # Groq LLM answer generation
│       └── job_manager.py     # Background ingestion job queue
├── docs/
│   └── design_decisions.md   # Mandatory explanation document
├── uploads/                   # Uploaded files (gitignored)
└── faiss_store/               # Persisted FAISS index (gitignored)
```

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `POST /upload` | 10 requests/minute per IP |
| `POST /query` | 30 requests/minute per IP |

---

## Design Decisions

See [`docs/design_decisions.md`](docs/design_decisions.md) for:
- Chunk size rationale (why 512 characters)
- Observed retrieval failure case (multi-hop questions)
- Latency metric tracking and observations
- Why FAISS over Pinecone
- Why `all-MiniLM-L6-v2` over OpenAI embeddings
- Why not LangChain

---

## Extending the System

| Feature | Where to change |
|---------|----------------|
| Add DOCX support | `app/services/parser.py` |
| Switch to Pinecone | `app/services/vector_store.py` |
| Add streaming LLM responses | `app/services/llm.py` + route |
| Persistent job store | `app/services/job_manager.py` → swap dict for Redis |
| Add auth | FastAPI `Depends` on route functions |
