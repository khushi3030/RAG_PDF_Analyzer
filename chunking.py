"""
chunking.py
-----------
Recursive character text splitter.

Why "recursive"? Instead of blindly cutting text every N characters (which
slices sentences and words in half), this splitter tries a list of
separators from "most semantic" to "least semantic":

    ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

It first tries to split on paragraph breaks. If a resulting piece is still
bigger than chunk_size, it recurses into that piece using the NEXT
separator (single newline), then sentence boundaries, then words, then
finally raw characters as a last resort. This keeps chunks as semantically
coherent as possible while still respecting a hard size limit.

Overlap: consecutive chunks share `chunk_overlap` characters so that
context isn't lost right at a chunk boundary (e.g. a sentence that
explains a term defined at the end of the previous chunk).
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        return [c for c in self._split(text, self.separators) if c.strip()]

    def split_documents(self, pages: List[dict]) -> List[Chunk]:
        """
        pages: list of {"text": str, "page_number": int, "source": str}
        Returns a flat list of Chunk objects carrying page/source metadata,
        which is what lets the chatbot later cite "page 4 of report.pdf".
        """
        chunks: List[Chunk] = []
        for page in pages:
            pieces = self.split_text(page["text"])
            for i, piece in enumerate(pieces):
                chunks.append(
                    Chunk(
                        text=piece,
                        metadata={
                            "source": page.get("source", "unknown"),
                            "page_number": page.get("page_number"),
                            "chunk_index": i,
                        },
                    )
                )
        return chunks

    # ---- internals ----------------------------------------------------

    def _split(self, text: str, separators: List[str]) -> List[str]:
        separator = separators[-1]
        remaining_separators: List[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                remaining_separators = separators[i + 1:]
                break

        splits = list(text) if separator == "" else text.split(separator)
        splits = [s for s in splits if s != ""]

        final_chunks: List[str] = []
        good_splits: List[str] = []

        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    final_chunks.extend(self._merge(good_splits, separator))
                    good_splits = []
                if remaining_separators:
                    final_chunks.extend(self._split(s, remaining_separators))
                else:
                    final_chunks.append(s)

        if good_splits:
            final_chunks.extend(self._merge(good_splits, separator))

        return final_chunks

    def _merge(self, splits: List[str], separator: str) -> List[str]:
        """Greedily pack small splits into chunks close to chunk_size,
        carrying `chunk_overlap` characters of context into the next chunk."""
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for s in splits:
            piece_len = len(s) + (len(separator) if current else 0)
            if current_len + piece_len > self.chunk_size and current:
                chunks.append(separator.join(current))
                # slide the window: drop from the front until we're within overlap
                while current_len > self.chunk_overlap and len(current) > 1:
                    dropped = current.pop(0)
                    current_len -= len(dropped) + len(separator)
            current.append(s)
            current_len += piece_len

        if current:
            chunks.append(separator.join(current))

        return chunks