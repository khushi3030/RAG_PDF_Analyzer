A fully local, privacy-first RAG (Retrieval-Augmented Generation) system for asking natural-language questions about your own PDFs — no data ever leaves your machine. Powered by Ollama (LLM, embeddings, and optional vision model), ChromaDB for persistent vector storage, and a cross-encoder reranker for retrieval precision.

Beyond basic "embed and search," this project implements two-stage retrieval with reranking, per-document retrieval scoping, optional multimodal ingestion, prompt-injection defenses, corrective/self-checking generation, an analytics dashboard, and a standalone evaluation harness.

Features
Multi-document PDF chat with retrieval scoping — upload multiple PDFs and select exactly which document(s) a question should search, instead of every query hitting your entire indexed collection.
Two-stage retrieval — fast bi-encoder cosine similarity search (ChromaDB + Ollama embeddings) followed by cross-encoder reranking (sentence-transformers) for precision.
Corrective / self-checking RAG — after generating an answer, the model grades whether it's actually supported by the retrieved context and automatically rewrites the query and retries if not.
Prompt-injection defenses — regex-based redaction of instruction-like phrases in document content, plus explicit <document_content> delimiting so untrusted PDF text is never confused with system instructions.
Optional multimodal ingestion — OCR on scanned pages, and vision-model description of embedded figures and vector-drawn charts (via a local vision model), toggle-able since it's the slowest ingestion step.
Persistent, on-disk vector store — documents stay indexed across app restarts; browse, delete, or clear indexed content from the UI.
Analytics dashboard — per-document chunk/page stats, a PCA projection of the embedding space, and query latency history.
Standalone evaluation harness — measure retrieval hit-rate, LLM-as-judge answer correctness, and per-stage latency against a hand-written, ground-truthed question set — so pipeline changes can be measured, not guessed at.


Getting Started
Option A: Local (Python + Ollama)

1. Install Ollama and pull the required models:

bash
ollama pull llama3.2
ollama pull nomic-embed-text
ollama pull llava        # optional — only needed for image/scanned-page analysis

2. Set up a Python environment and install dependencies:

bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

3. Run the app:

bash
streamlit run app.py

The app opens at http://localhost:8501. Indexed documents persist to ./chroma_db on disk.


Usage
Upload PDFs in the sidebar and click Process documents. Adjust chunk size/overlap, retrieval depth, and models in the Settings panel first if needed.
Ask questions in the Chat tab. Use the "Ask about" selector to scope a question to one or more specific documents rather than searching everything indexed.
Enable corrective mode (sidebar checkbox) if you want the model to self-check and retry ungrounded answers — slower, since it can't stream, but more reliable.
Check the Dashboard tab for per-document stats, an embedding-space visualization, and query history — both respect the same document scope selected in Chat.
