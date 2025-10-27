create or replace function public.match_documents(
  query_embedding vector,
  match_count int default 5
)
returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language sql stable as $$
  select d.id, d.content, d.metadata,
         1 - (d.embedding <=> query_embedding) as similarity
  from public.documents d
  order by d.embedding <=> query_embedding
  limit match_count;
$$;
