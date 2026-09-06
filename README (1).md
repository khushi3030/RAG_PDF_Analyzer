# 📄 Local PDF RAG Chatbot

A fully **local**, privacy-first RAG (Retrieval-Augmented Generation) system for asking natural-language questions about your own PDFs — no data ever leaves your machine. Powered by **Ollama** (LLM, embeddings, and optional vision model), **ChromaDB** for persistent vector storage, and a **cross-encoder reranker** for retrieval precision.

Beyond basic "embed and search," this project implements two-stage retrieval with reranking, per-document retrieval scoping, optional multimodal ingestion, prompt-injection defenses, corrective/self-checking generation, an analytics dashboard, and a standalone evaluation harness.

> 📐 For a deep dive into every component and the full request/response flow, see [`PROJECT_ARCHITECTURE.md`](./PROJECT_ARCHITECTURE.md).

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Option A: Local (Python + Ollama)](#option-a-local-python--ollama)
  - [Option B: Docker](#option-b-docker)
- [Usage](#usage)
- [Evaluation Harness](#evaluation-harness)
- [Configuration](#configuration)
- [Known Limitations & Roadmap](#known-limitations--roadmap)
- [License](#license)

---

## Features

- **Multi-document PDF chat with retrieval scoping** — upload multiple PDFs and select exactly which document(s) a question should search, instead of every query hitting your entire indexed collection.
- **Two-stage retrieval** — fast bi-encoder cosine similarity search (ChromaDB + Ollama embeddings) followed by cross-encoder reranking (`sentence-transformers`) for precision.
- **Corrective / self-checking RAG** — after generating an answer, the model grades whether it's actually supported by the retrieved context and automatically rewrites the query and retries if not.
- **Prompt-injection defenses** — regex-based redaction of instruction-like phrases in document content, plus explicit `<document_content>` delimiting so untrusted PDF text is never confused with system instructions.
- **Optional multimodal ingestion** — OCR on scanned pages, and vision-model description of embedded figures and vector-drawn charts (via a local vision model), toggle-able since it's the slowest ingestion step.
- **Persistent, on-disk vector store** — documents stay indexed across app restarts; browse, delete, or clear indexed content from the UI.
- **Analytics dashboard** — per-document chunk/page stats, a PCA projection of the embedding space, and query latency history.
- **Standalone evaluation harness** — measure retrieval hit-rate, LLM-as-judge answer correctness, and per-stage latency against a hand-written, ground-truthed question set — so pipeline changes can be measured, not guessed at.
- **Dockerized** for reproducible local deployment.

---

## Tech Stack

| Layer | Tools |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| LLM / Embeddings / Vision | [Ollama](https://ollama.com) (`llama3.2`, `nomic-embed-text`, `llava`) |
| Vector Store | [ChromaDB](https://www.trychroma.com) (persistent, cosine similarity) |
| Reranking | `sentence-transformers` cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| PDF Parsing | [PyMuPDF](https://pymupdf.readthedocs.io) |
| Analytics | `numpy` (custom PCA), `pandas` |
| Deployment | Docker, Docker Compose |

---

## Project Structure

```
.
├── app.py                  # Streamlit UI — sidebar, chat, dashboard
├── pdf_processor.py         # Text extraction (PyMuPDF)
├── image_processor.py       # Optional: OCR + vision-model figure/chart description
├── chunking.py               # Recursive character text splitter
├── vector_store.py          # ChromaDB wrapper (embed, search, filter, delete)
├── reranker.py               # Cross-encoder reranking
├── rag_pipeline.py          # Orchestration: retrieve -> rerank -> sanitize -> generate
├── analytics.py               # Query logging, document stats, PCA projection
├── eval.py                    # Standalone evaluation harness (CLI)
├── eval_set.example.json    # Template for your own ground-truthed eval questions
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .streamlit/
│   └── config.toml          # Theme (dark ink background, amber accent)
└── PROJECT_ARCHITECTURE.md  # Full architecture & data-flow documentation
```

---

## Getting Started

### Option A: Local (Python + Ollama)

**1. Install [Ollama](https://ollama.com) and pull the required models:**

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
ollama pull llava        # optional — only needed for image/scanned-page analysis
```

**2. Set up a Python environment and install dependencies:**

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Run the app:**

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Indexed documents persist to `./chroma_db` on disk.

### Option B: Docker

Two containers — one for Ollama, one for the Streamlit app — wired together via `docker-compose.yml`.

```bash
docker compose up -d --build

# Pull models into the Ollama container (first run only)
docker exec -it rag-ollama ollama pull llama3.2
docker exec -it rag-ollama ollama pull nomic-embed-text
docker exec -it rag-ollama ollama pull llava
```

Open `http://localhost:8501`. Indexed documents and pulled models persist in named Docker volumes across restarts (`docker compose down` keeps them; `docker compose down -v` wipes them).

> **Note:** This setup targets local use and demos, not public cloud hosting — running Ollama with multiple models needs more RAM than most free-tier hosts provide.

---

## Usage

1. **Upload PDFs** in the sidebar and click **Process documents**. Adjust chunk size/overlap, retrieval depth, and models in the Settings panel first if needed.
2. **Ask questions** in the Chat tab. Use the **"Ask about"** selector to scope a question to one or more specific documents rather than searching everything indexed.
3. **Enable corrective mode** (sidebar checkbox) if you want the model to self-check and retry ungrounded answers — slower, since it can't stream, but more reliable.
4. **Check the Dashboard tab** for per-document stats, an embedding-space visualization, and query history — both respect the same document scope selected in Chat.

---

## Evaluation Harness

`eval.py` is a standalone CLI script — it runs against whatever is already indexed in `chroma_db` and is not part of the Streamlit app.

**1. Write your eval set** (copy the template and fill in real questions against your own PDFs):

```bash
cp eval_set.example.json eval_set.json
```

Each entry:
```json
{
  "id": "unique-label",
  "question": "A question about your PDF",
  "expected_source": "filename.pdf",
  "reference_answer": "The correct answer, in your own words"
}
```

**2. Run it:**

```bash
python eval.py --eval-set eval_set.json
```

**3. Read the output** — a printed summary plus a full `eval_results.json`:

```
=== Evaluation Summary ===
Questions evaluated:   15
Retrieval hit-rate:    86.7%
Answer correctness:    80.0%
Avg total latency:     2.14s
```

Run it before and after any pipeline change (chunk size, retrieval depth, a new reranker, hybrid search, etc.) to measure whether the change actually helped.

> **Important:** `eval.py` and `streamlit run app.py` must be run from the **same working directory** — both resolve `./chroma_db` relative to the current folder, so running them from different locations silently creates two separate databases.

---

## Configuration

All of the following are adjustable from the sidebar at runtime:

| Setting | Default | Notes |
|---|---|---|
| Chunk size | 1000 chars | Larger chunks preserve more context per chunk, at the cost of retrieval granularity |
| Chunk overlap | 150 chars | Prevents context loss at chunk boundaries |
| Chunks retrieved (bi-encoder) | 20 | Broad candidate pool before reranking |
| Chunks kept after rerank | 5 | Final context sent to the LLM |
| Ollama LLM model | `llama3.2` | Any locally pulled Ollama chat model |
| Ollama embedding model | `nomic-embed-text` | **Do not change after indexing** — mixing embedding models in one collection breaks similarity search |
| Corrective RAG | off | Self-checking + retry; disables streaming when on |
| Image/scanned-page analysis | off | Enables OCR + vision-model figure description; the slowest ingestion step |

---

## Known Limitations & Roadmap

- **Pure vector retrieval** — no BM25/keyword search yet, so exact terms (names, codes, numbers) that embeddings sometimes blur aren't specifically caught. Hybrid search (vector + BM25 via reciprocal rank fusion) is a planned improvement.
- **No conversational query rewriting** — retrieval only searches the literal current question, so vague follow-ups referencing prior turns may retrieve poorly.
- **No table-aware extraction** — PyMuPDF's plain text extraction can mangle tabular data; `pdfplumber` would help on data-heavy PDFs.
- **Sequential embedding on ingestion** — chunks are embedded one Ollama call at a time.
- **Not deployable to free-tier public cloud hosting as-is** — the local-LLM architecture needs more RAM than free hosting tiers typically provide.

See [`PROJECT_ARCHITECTURE.md`](./PROJECT_ARCHITECTURE.md) for the full design-decisions rationale behind the current architecture.

---

## License

Add a license of your choice (e.g., MIT) here before publishing.
