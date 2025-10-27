from typing import List
from backend.config import Settings
from backend.ingest.loaders import load_from_urls
from backend.common.vectorstore import get_vectorstore

def ingest_urls(urls: List[str], settings: Settings):
    docs = load_from_urls(urls, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    vs = get_vectorstore(settings)
    return vs.add_documents(docs)
