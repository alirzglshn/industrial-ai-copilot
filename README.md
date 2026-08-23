# Industrial AI Copilot

An AI engineering copilot for technical manuals. Upload a PDF (text, tables,
diagrams, images) and ask questions about the equipment it describes — the
system retrieves relevant text and images and answers with citations back to
the source page, using a local VLM/LLM, refusing to answer when the evidence
isn't there.

Status: **Phase 2 complete — document ingestion pipeline.** Uploading a PDF
now parses it page by page (text, ruled tables, embedded diagrams), chunks the
text, and persists everything to Postgres with page numbers preserved end to
end. Retrieval and answering are not built yet, so `/query` returns `501`.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and the full roadmap.

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
├── retrieval/base.py     Retriever interface              (implemented in Phase 3-4)
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
pip install -e .
uvicorn copilot.main:app --reload
```

You'll need a local Postgres and Qdrant reachable at the URLs in `.env`
(the `docker compose up postgres qdrant` subset works for this).

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Roadmap

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full 8-phase plan (ingestion,
embeddings/vector search, multimodal retrieval, grounded generation, the
tool-using agent layer, evaluation, and productionization).
