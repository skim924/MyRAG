import json
import math
import re
from typing import List, Dict, Any
from uuid import uuid4

import requests

from backend.config import Settings

def _sb_headers(settings: Settings, use_service_key: bool = True) -> Dict[str, str]:
    key = settings.supabase_service_role_key if use_service_key else settings.supabase_anon_key
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _simple_features(text: str) -> List[float]:
    tokens = _tokenize(text)
    if not tokens:
        return [0.0, 0.0, 0.0]
    length = float(len(text))
    word_count = float(len(tokens))
    unique_words = float(len(set(tokens)))
    return [length, word_count, unique_words]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def embed_query(text: str, settings: Settings) -> List[float]:
    if (settings.vectorstore_backend or "").lower() == "memory" or not settings.supabase_url:
        return _simple_features(text)

    url = f"{settings.ollama_host}/api/embeddings"
    payload = {"model": settings.embeddings_model, "prompt": text}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    vec = r.json().get("embedding")
    if not isinstance(vec, list):
        raise RuntimeError("Ollama embedding response invalid")
    return vec

def embed_texts(texts: List[str], settings: Settings) -> List[List[float]]:
    if (settings.vectorstore_backend or "").lower() == "memory" or not settings.supabase_url:
        return [_simple_features(t) for t in texts]

    url = f"{settings.ollama_host}/api/embeddings"
    out = []
    for t in texts:
        r = requests.post(url, json={"model": settings.embeddings_model, "prompt": t}, timeout=60)
        r.raise_for_status()
        out.append(r.json().get("embedding"))
    return out

class SupabaseVectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.supabase_url:
            raise ValueError("SUPABASE_URL is not set")
        if not (settings.supabase_anon_key or settings.supabase_service_role_key):
            raise ValueError("Supabase key is not set")

    def add_documents(self, docs: List[Dict[str, Any]]):
        contents = [d["content"] for d in docs]
        embeddings = embed_texts(contents, self.settings)
        rows = []
        for d, vec in zip(docs, embeddings):
            rows.append({
                "content": d["content"],
                "metadata": d.get("metadata", {}),
                "embedding": vec
            })
        url = f"{self.settings.supabase_url}/rest/v1/{self.settings.documents_table}"
        r = requests.post(url, headers=_sb_headers(self.settings, use_service_key=True), data=json.dumps(rows), timeout=120)
        r.raise_for_status()
        return r.json()

    def match(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        url = f"{self.settings.supabase_url}/rest/v1/rpc/{self.settings.match_rpc}"
        payload = {"query_embedding": query_embedding, "match_count": top_k}
        r = requests.post(url, headers=_sb_headers(self.settings, use_service_key=False), data=json.dumps(payload), timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Supabase RPC error {r.status_code}: {r.text}")
        return r.json()

def get_vectorstore(settings: Settings) -> 'SupabaseVectorStore':
    backend = (settings.vectorstore_backend or "supabase").lower()
    if backend == "memory" or not settings.supabase_url:
        return InMemoryVectorStore.get_instance(settings)
    return SupabaseVectorStore(settings)


class InMemoryVectorStore:
    _instance: "InMemoryVectorStore" = None

    def __init__(self, settings: Settings):
        self.settings = settings
        self._rows: List[Dict[str, Any]] = []

    @classmethod
    def get_instance(cls, settings: Settings) -> "InMemoryVectorStore":
        if cls._instance is None:
            cls._instance = cls(settings)
        elif cls._instance.settings is not settings:
            # Refresh settings reference so updated env vars are reflected
            cls._instance.settings = settings
        return cls._instance

    @classmethod
    def reset(cls):
        if cls._instance is not None:
            cls._instance._rows.clear()

    def add_documents(self, docs: List[Dict[str, Any]]):
        contents = [d["content"] for d in docs]
        embeddings = embed_texts(contents, self.settings)
        new_rows = []
        for doc, emb in zip(docs, embeddings):
            row = {
                "id": str(uuid4()),
                "content": doc["content"],
                "metadata": doc.get("metadata", {}) or {},
                "embedding": emb,
                "similarity": None,
            }
            self._rows.append(row)
            new_rows.append(row)
        return new_rows

    def match(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        scored = []
        for row in self._rows:
            sim = _cosine_similarity(query_embedding, row["embedding"])
            scored.append((sim, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for sim, row in scored[:top_k]:
            out.append({
                "id": row["id"],
                "content": row["content"],
                "metadata": row.get("metadata", {}),
                "similarity": sim
            })
        return out
