"""
reranker.py
-----------
Why rerank at all?

The vector store's cosine-similarity search (a "bi-encoder" retrieval —
query and chunk are embedded separately, then compared) is fast and scales
to huge corpora, but it's approximate: it compares two independently
computed vectors and can miss subtle relevance signals, especially when
the query and the correct chunk phrase the same idea very differently.

A cross-encoder reads the query and a candidate chunk TOGETHER, in a
single forward pass, and directly outputs a relevance score. This is far
more accurate but too slow to run over an entire document collection —
so the standard pattern (used here) is:

    1. Bi-encoder (cosine similarity) retrieves a broad candidate set,
       e.g. top 20, cheaply.
    2. Cross-encoder reranks just those 20 candidates, expensively but
       accurately, and we keep the top 5 to send to the LLM.

This two-stage "retrieve then rerank" pipeline is what meaningfully
improves answer quality over cosine similarity alone.
"""

from typing import List, Dict
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Downloaded once from HuggingFace on first run, then cached locally.
        # This model itself is small (~80MB) and CPU-friendly.
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        if not candidates:
            return []
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k]