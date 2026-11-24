"""Chunking, persistent storage and hybrid retrieval of legal sources."""
from __future__ import annotations
import json, logging, re
from dataclasses import dataclass, field
from pathlib import Path
from .config import CHUNK_OVERLAP, CHUNK_SIZE, STORE_FILE, TOP_K_RESULTS, MIN_RELEVANCE_SCORE, Config, OllamaUnavailableError
from .ollama_client import OllamaClient
LOGGER = logging.getLogger("gambia_law_chatbot.store")

@dataclass
class Chunk:
    source: str
    source_url: str
    chunk_index: int
    text: str
    embedding: list[float] | None = field(default=None)
    document_type: str = "Unverified document type"
    publication_date: str = "Not stated"
    def to_dict(self) -> dict: return self.__dict__.copy()
    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        allowed = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**allowed)

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size: return [text] if text.strip() else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end))
            if boundary > start + size // 2: end = boundary
        chunks.append(text[start:end].strip())
        if end >= len(text): break
        start = max(start + 1, end - overlap)
    return [c for c in chunks if c]

def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b): return 0.0
    dot = sum(x*y for x,y in zip(a,b)); na = sum(x*x for x in a)**.5; nb = sum(x*x for x in b)**.5
    return dot/(na*nb) if na and nb else 0.0

def terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in {"the","and","for","that","with","from","what","does","under","about","gambia"}}
def keyword_score(query: str, chunk: Chunk) -> float:
    q = terms(query); body = terms(chunk.text + " " + chunk.source)
    if not q: return 0.0
    overlap = len(q & body) / len(q)
    phrase_bonus = .15 if query.lower() in (chunk.source + " " + chunk.text).lower() else 0
    return min(1.0, overlap + phrase_bonus)

class LawStore:
    def __init__(self, path: Path = STORE_FILE) -> None:
        self.path = path; self.chunks: list[Chunk] = []; self._seen: set[tuple[str,int]] = set()
        if path.exists():
            try: self.chunks = [Chunk.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]
            except (json.JSONDecodeError, TypeError, KeyError) as exc: LOGGER.warning("Could not load index: %s", exc)
        self._seen = {(c.source_url,c.chunk_index) for c in self.chunks}
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp"); tmp.write_text(json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False), encoding="utf-8"); tmp.replace(self.path)
    def has_chunk(self, url: str, index: int) -> bool: return (url,index) in self._seen
    def add(self, chunk: Chunk) -> None:
        self.chunks.append(chunk); self._seen.add((chunk.source_url,chunk.chunk_index))
    def retrieve(self, query: str, config: Config, k: int = TOP_K_RESULTS) -> list[Chunk]:
        if not self.chunks: return []
        query_vector = None
        try: query_vector = OllamaClient(config).embed(query)
        except OllamaUnavailableError: LOGGER.warning("Embeddings unavailable; using keyword retrieval.")
        ranked = []
        for chunk in self.chunks:
            lexical = keyword_score(query, chunk)
            semantic = cosine(query_vector, chunk.embedding or []) if query_vector else 0.0
            score = (0.68 * semantic + 0.32 * lexical) if query_vector and chunk.embedding else lexical
            ranked.append((score, chunk))
        ranked.sort(key=lambda x:x[0], reverse=True)
        # Return only genuinely relevant evidence, rather than forcing unrelated top-k excerpts.
        return [c for score,c in ranked[:k] if score >= MIN_RELEVANCE_SCORE]
    @property
    def status(self) -> str: return f"Indexed {len(self.chunks)} searchable passages from {len({c.source_url for c in self.chunks})} document(s)." if self.chunks else "No documents indexed yet."
