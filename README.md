# Industrial AI Copilot

An AI engineering copilot for technical manuals. Upload a PDF (text, tables,
diagrams, images) and ask questions about the equipment it describes — the
system retrieves relevant text and images and answers with citations back to
the source page, using a local VLM/LLM, refusing to answer when the evidence
isn't there.

Status: **Phase 3 complete — semantic search.** Uploading a PDF parses it page
by page (text, ruled tables, embedded diagrams), chunks the text, embeds it
with a local model, and indexes it in Qdrant. `POST /search` answers questions
with the passages that match and the page each came from. Answer generation is
not built yet, so `/query` returns `501`. See [ARCHITECTURE.md](ARCHITECTURE.md)
for the design and the full roadmap.

Everything runs locally and free: open-weight models, no paid APIs, no keys.

## Stack

Python · FastAPI · PyTorch/Hugging Face Transformers · Qdrant · PostgreSQL ·
Docker

## Project layout

```
src/copilot/
├── main.py             FastAPI app factory
├── core/                settings (env-driven) + logging
├── api/routes/          health, documents, query
├── db/                   SQLAlchemy models (Document, Page, Chunk, Image) + session
├── schemas/              Pydantic request/response models
├── ingestion/            DONE — PDF -> text/tables/images -> chunks -> Postgres
│   ├── parser.py           pdfplumber (text + tables) + pypdf (images)
│   ├── chunker.py          page-scoped, structure-aware, overlapping chunks
│   └── service.py          parse -> chunk -> persist orchestration
├── retrieval/            DONE (text) — embeddings, Qdrant, semantic search
│   ├── embedder.py         sentence-transformers, BGE query prefix
│   ├── vector_store.py     Qdrant collection, upsert, filtered search
│   ├── indexer.py          chunks -> vectors
│   └── retriever.py        query -> Evidence with page numbers
├── generation/base.py    AnswerGenerator interface         (implemented in Phase 5)
└── agent/base.py         Tool / Agent interfaces            (implemented in Phase 6)
```

## Ingesting a manual

```bash
curl -F "file=@manual.pdf" http://localhost:8000/documents/upload
```

```json
{
  "id": "0a1b…",
  "filename": "manual.pdf",
  "status": "parsed",
  "page_count": 42,
  "chunk_count": 187,
  "image_count": 23
}
```

Inspect what was extracted:

```bash
curl http://localhost:8000/documents                        # list manuals
curl http://localhost:8000/documents/{id}/chunks            # all text chunks
curl "http://localhost:8000/documents/{id}/chunks?page_number=37"
curl http://localhost:8000/documents/{id}/images            # extracted diagrams
```

Every chunk carries its `document_id` and `page_number`, which is what makes
the page-level citations in later phases possible.

## Searching

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "why is the pump overheating?", "top_k": 3}'
```

```json
{
  "query": "why is the pump overheating?",
  "results": [
    {
      "kind": "text",
      "document_id": "0a1b…",
      "page_number": 37,
      "score": 0.746,
      "text": "Overheating is most commonly caused by insufficient cooling airflow…"
    }
  ]
}
```

Add `"document_id"` to search within one manual. Uploading indexes
automatically; `POST /documents/{id}/index` re-indexes (safe to repeat).

If the embedding model or Qdrant is unavailable, uploads still succeed and
report `indexed_chunks: 0` — index later rather than losing the ingest.

## Tests

```bash
pytest                  # 79 unit tests, no model download
pytest -m integration   # 6 more, using the real embedding model
```

The unit suite runs Qdrant in-process and swaps in a lexical stand-in for the
embedder, so it needs neither a container nor model weights. The integration
suite loads the real model and asserts genuinely semantic behaviour.

## Running locally

Requirements: Docker Desktop.

```bash
cp .env.example .env
docker compose up --build
```

This starts:

- `api` — FastAPI app on http://localhost:8000 (docs at `/docs`)
- `postgres` — metadata store on port 5432
- `qdrant` — vector store on ports 6333 (HTTP) / 6334 (gRPC)

Check it's alive:

```bash
curl http://localhost:8000/health
```

## Running the API without Docker

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[ai,dev]" --extra-index-url https://download.pytorch.org/whl/cpu
uvicorn copilot.main:app --reload
```

The `ai` extra pulls torch and sentence-transformers, needed for embeddings.
The CPU wheel index keeps it from downloading multi-gigabyte CUDA builds that
are useless without a dedicated GPU. Without the extra the API still runs and
ingests PDFs; only indexing and search are unavailable.

You'll need a local Postgres and Qdrant reachable at the URLs in `.env`
(the `docker compose up postgres qdrant` subset works for this).


## Roadmap

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full 8-phase plan (ingestion,
embeddings/vector search, multimodal retrieval, grounded generation, the
tool-using agent layer, evaluation, and productionization).
