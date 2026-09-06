
"""
analytics.py
-------------
Lightweight, dependency-free analytics backing the Dashboard tab:
 
  - Per-query logging (latency, number of sources, corrective-RAG verdict),
    persisted to a local JSON file so history survives app restarts.
  - Document-level stats (chunk/page counts per source).
  - A 2D projection of chunk embeddings via numpy-only PCA, so you can
    visually see how chunks cluster by document/topic — no sklearn/umap
    dependency needed for this.
"""
 
import json
import time
from pathlib import Path
from typing import List, Dict
 
import numpy as np
 
QUERY_LOG_PATH = "./query_log.json"
 
 
def log_query(entry: Dict, path: str = QUERY_LOG_PATH) -> None:
    log = load_query_log(path)
    log.append(entry)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
 
 
def load_query_log(path: str = QUERY_LOG_PATH) -> List[Dict]:
    if not Path(path).exists():
        return []
    with open(path) as f:
        return json.load(f)
 
 
def new_query_entry(question: str, latency_sec: float, num_sources: int, supported) -> Dict:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "latency_sec": round(latency_sec, 2),
        "num_sources": num_sources,
        # None when corrective mode is off (no grading happened),
        # otherwise True/False from the self-check step.
        "supported": supported,
    }
 
 
def document_stats(metadatas: List[dict]) -> Dict[str, Dict]:
    """metadatas: the full list of chunk metadata dicts from VectorStore.get_all()."""
    stats: Dict[str, Dict] = {}
    for meta in metadatas:
        src = meta.get("source", "unknown")
        stats.setdefault(src, {"chunks": 0, "pages": set()})
        stats[src]["chunks"] += 1
        stats[src]["pages"].add(meta.get("page_number"))
    return {src: {"chunks": v["chunks"], "pages": len(v["pages"])} for src, v in stats.items()}
 
 
def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """
    Numpy-only PCA to project high-dimensional chunk embeddings down to 2D
    for visualization — avoids adding sklearn/umap as a dependency just for
    one chart.
 
    Standard approach: center the data, then project onto the top-2
    principal components (the directions of greatest variance), found via
    SVD. For centered data X, X = U @ S @ Vt, and the rows of Vt are the
    principal directions ranked by explained variance (largest singular
    value first) — so projecting onto the first two rows of Vt gives the
    2D view that preserves the most variance possible in 2 dimensions.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] < 2:
        return np.zeros((embeddings.shape[0], 2))
    centered = embeddings - embeddings.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T
 
