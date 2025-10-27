import re
import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup

def fetch_url(url: str, timeout: int = 30) -> str:
    headers = { "User-Agent": "MyRAG/1.0 (+https://example.com)" }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    i = 0
    step = chunk_size - chunk_overlap
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        i += step
    return chunks

def load_from_urls(urls: List[str], chunk_size: int = 1200, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
    docs = []
    for u in urls:
        try:
            html = fetch_url(u)
            text = html_to_text(html)
            for ch in chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
                docs.append({"content": ch, "metadata": {"source": u}})
        except Exception as e:
            print(f"[loader] Failed to fetch {u}: {e}")
    return docs
