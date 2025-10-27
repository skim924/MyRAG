# MyRAG (Full Project)

Text-only RAG with FastAPI backend, Supabase pgvector, and Ollama (chat + embeddings).

## Backend
```
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in SUPABASE_* and models if needed
uvicorn backend.app.main:app --reload
```

## Supabase
- Table `documents(id uuid pk default gen_random_uuid(), content text, metadata jsonb, embedding vector(1024))`
- IVFFLAT index on `embedding`
- RPC from `supabase/0001_create_match_documents.sql`

## Frontend
```
cd frontend
npm install
npm run dev
```
Open http://localhost:5173/

## Notes
- We call PostgREST directly via `requests`, avoiding supabase python client conflicts.
- Adjust `embedding vector(DIM)` to match your Ollama embedding model. For `nomic-embed-text` use 768 or 1024 depending on model; if 768, alter the column and reindex.
