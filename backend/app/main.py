from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.config import Settings
from backend.common.vectorstore import get_vectorstore, embed_query
from backend.prompts.answer_template import SYSTEM_TEMPLATE, USER_TEMPLATE

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

settings = Settings()
app = FastAPI(title="MyRAG API", version="0.2.0")

allowed_origins = [o.strip() for o in (settings.cors_allow_origins or "").split(",") if o.strip()]
if not allowed_origins:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOllama(model=settings.chat_model, base_url=settings.ollama_host, temperature=0.1)

class IngestRequest(BaseModel):
    urls: List[str]

class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    with_answer: bool = True
    chat_history: Optional[List[Dict[str, str]]] = None
    sources_filter: Optional[List[str]] = None

class QueryResultChunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity: Optional[float] = None

class QueryResponse(BaseModel):
    answer: Optional[str] = None
    results: List[QueryResultChunk]
    used_model: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok", "ollama": settings.ollama_host}

@app.post("/ingest")
def ingest(req: IngestRequest):
    from backend.ingest.ingest import ingest_urls
    try:
        res = ingest_urls(req.urls, settings=settings)
        return {"inserted": len(res)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")

def _format_context_and_sources(chunks: List[Dict[str, Any]], k: int, max_chars: int = 3500):
    acc, char_count = [], 0
    for i, c in enumerate(chunks[:k], start=1):
        snippet = c["content"].strip().replace("\n", " ")
        if char_count + len(snippet) > max_chars:
            break
        acc.append(f"[{i}] {snippet}")
        char_count += len(snippet)
    context = "\n\n".join(acc)

    srcs = []
    for i, c in enumerate(chunks[:k], start=1):
        meta = c.get("metadata") or {}
        src = meta.get("source") or meta.get("url") or "unknown"
        srcs.append(f"[{i}] {src}")
    sources_list = "\n".join(srcs)
    return context, sources_list

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        vs = get_vectorstore(settings=settings)
        q_emb = embed_query(req.query, settings=settings)
        rows = vs.match(q_emb, top_k=req.top_k)

        if req.sources_filter:
            allowed = set(req.sources_filter)
            rows = [
                r for r in rows
                if (r.get("metadata") or {}).get("source") in allowed
            ]

        results = [
            QueryResultChunk(
                id=str(r.get("id")),
                content=r.get("content", ""),
                metadata=r.get("metadata", {}) or {},
                similarity=r.get("similarity")
            ) for r in rows
        ]

        if not req.with_answer:
            return QueryResponse(answer=None, results=results, used_model=settings.chat_model)

        if not results:
            return QueryResponse(answer="No relevant documents were found.", results=[], used_model=settings.chat_model)

        context, sources_list = _format_context_and_sources([r.model_dump() for r in results], req.top_k)

        messages = [SystemMessage(content=SYSTEM_TEMPLATE)]
        if req.chat_history:
            for m in req.chat_history:
                role = m.get("role")
                txt = m.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=txt))
                elif role == "assistant":
                    messages.append(SystemMessage(content=f"[Previous assistant]\n{txt}"))
        messages.append(HumanMessage(content=USER_TEMPLATE.format(
            question=req.query, k=req.top_k, context=context, sources_list=sources_list, max_tokens=512
        )))

        answer = llm.invoke(messages).content.strip()
        return QueryResponse(answer=answer, results=results, used_model=settings.chat_model)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
