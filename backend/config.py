import os

class Settings:
    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    # Ollama
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    chat_model: str = os.getenv("CHAT_MODEL", "llama3.2")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
    # Supabase
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_anon_key: str = os.getenv("SUPABASE_ANON_KEY", "")
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    # DB objects
    documents_table: str = os.getenv("DOCUMENTS_TABLE", "documents")
    match_rpc: str = os.getenv("MATCH_RPC", "match_documents")
    vectorstore_backend: str = os.getenv("VECTORSTORE_BACKEND", "supabase")
    cors_allow_origins: str = os.getenv("CORS_ALLOW_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
    # Other
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
