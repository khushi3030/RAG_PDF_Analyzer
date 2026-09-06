"""
vector_store.py
----------------
ChromaDB-backed vector store (replaces the earlier plain-numpy version).

Why ChromaDB: it's an embedded, zero-config vector database — no separate
server process, just a local folder on disk — that gives us three things
the numpy version didn't have for free:

  1. Persistence: chunks survive an app restart instead of living only in
     RAM. Point two runs at the same persist_dir and the second run sees
     everything the first one indexed.
  2. Built-in cosine similarity search via an HNSW index, instead of the
     linear numpy scan we implemented by hand.
  3. Native metadata storage/filtering alongside each vector (used here to
     delete or browse chunks by source filename).

We still generate embeddings ourselves via Ollama rather than using
Chroma's default embedding function, so the embedding model chosen in the
Streamlit sidebar is respected end to end, same as before.
"""

import hashlib
from typing import List, Dict

import chromadb
import ollama

from chunking import Chunk

DEFAULT_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_chunks"


class VectorStore:
    def __init__(self, embed_model: str = "nomic-embed-text", persist_dir: str = DEFAULT_PERSIST_DIR):
        self.embed_model = embed_model
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        # "hnsw:space": "cosine" makes Chroma's index use cosine similarity
        # (its default is squared L2 distance, which we don't want here).
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ---- embedding ------------------------------------------------

    def embed_text(self, text: str) -> List[float]:
        response = ollama.embeddings(model=self.embed_model, prompt=text)
        return response["embedding"]

    # ---- building the index ----------------------------------------

    @staticmethod
    def _chunk_id(chunk: Chunk) -> str:
        """
        Stable, content-based ID: hashing source + page + chunk_index + text
        means re-processing the SAME pdf produces the SAME ids, so upsert()
        overwrites the existing entry instead of piling up duplicates.
        """
        meta = chunk.metadata
        raw = f"{meta.get('source')}|{meta.get('page_number')}|{meta.get('chunk_index')}|{chunk.text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def add_chunks(self, chunks: List[Chunk], progress_callback=None) -> None:
        if not chunks:
            return
        ids, embeddings, documents, metadatas = [], [], [], []
        for i, chunk in enumerate(chunks):
            ids.append(self._chunk_id(chunk))
            embeddings.append(self.embed_text(chunk.text))
            documents.append(chunk.text)
            metadatas.append(chunk.metadata)
            if progress_callback:
                progress_callback(i + 1, len(chunks))

        # upsert = add-or-overwrite by id — this is what makes re-processing
        # the same PDF safe instead of creating duplicate chunks.
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    # ---- retrieval --------------------------------------------------

    def search(self, query: str, top_k: int = 20, source_filter: List[str] = None) -> List[Dict]:
        """
        source_filter: optional list of source filenames to restrict the
        search to (e.g. only the PDF(s) currently selected in the UI).
        None or an empty list means "search everything" — pass the actual
        list of selected filenames to scope retrieval to those documents
        only, which is what keeps answers from bleeding across PDFs.
        """
        if self.collection.count() == 0:
            return []
        query_vec = self.embed_text(query)
        top_k = min(top_k, self.collection.count())

        where = None
        if source_filter:
            # Chroma's filter syntax for "source is one of these values"
            where = {"source": {"$in": source_filter}}

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        if not results["documents"][0]:
            return []
        output = []
        for doc, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            # Chroma's cosine "distance" = 1 - cosine_similarity, so convert
            # back to a similarity score for consistency with the rest of
            # the pipeline (higher = more relevant).
            output.append({"text": doc, "metadata": meta, "score": 1 - distance})
        return output

    # ---- browsing / inspection ---------------------------------------

    def peek(self, limit: int = 10) -> List[Dict]:
        """Returns a sample of stored chunks — used by the 'Browse indexed
        chunks' panel in app.py so you can see what's actually in the index,
        including chunks from documents processed in earlier sessions."""
        result = self.collection.get(limit=limit, include=["documents", "metadatas"])
        return [
            {"id": id_, "text": doc, "metadata": meta}
            for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]

    def get_all(self) -> Dict:
        """Returns everything in the collection, embeddings included — used
        by the dashboard for document stats and the embedding-space
        visualization. Fine at the scale of a local single-user app; a
        production version would paginate this instead of loading it all
        into memory at once."""
        if self.collection.count() == 0:
            return {"ids": [], "embeddings": [], "metadatas": [], "documents": []}
        result = self.collection.get(include=["embeddings", "metadatas", "documents"])
        return {
            "ids": result["ids"],
            "embeddings": result["embeddings"],
            "metadatas": result["metadatas"],
            "documents": result["documents"],
        }

    def list_sources(self) -> List[str]:
        """Returns the distinct set of source filenames currently indexed,
        including ones from previous sessions thanks to persistence."""
        if self.collection.count() == 0:
            return []
        result = self.collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in result["metadatas"]}
        return sorted(sources)

    def delete_source(self, source_name: str) -> None:
        """Removes all chunks belonging to one source file."""
        self.collection.delete(where={"source": source_name})

    def clear(self) -> None:
        """Wipes the entire collection from disk."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def __len__(self) -> int:
        return self.collection.count()