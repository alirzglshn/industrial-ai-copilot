# Industrial AI Copilot

An AI engineering copilot for technical manuals. Upload a PDF (text, tables,
diagrams, images) and ask questions about the equipment it describes — the
system retrieves relevant text and images and answers with citations back to
the source page, using a local VLM/LLM, refusing to answer when the evidence
isn't there.

Status: **Phase 8 complete — a full local app.** Uploading a PDF parses it page
by page (text, ruled tables, embedded diagrams), chunks the text, embeds both
text and images with local models, and indexes them in Qdrant. `POST /search`
returns the passages *and the diagrams* that match; `POST /query` answers a
question from that evidence with a local model, citing the pages it used and
refusing when the manual does not support an answer. `POST /agent/query` goes
further: an agent decides for itself whether a question needs document
search, image search, an exact page lookup, a calculation, or document
metadata, then answers from whatever it gathered — held to the exact same
grounding and faithfulness checks `/query` uses. A React UI sits in front of
all of it — streamed answers instead of a minute of blank screen, browsable
Q&A history, and click-through source-page previews on every citation. One
command runs the whole thing locally:

```bash
cp .env.example .env
docker compose up --build
```

then open **http://localhost:3000**. Nothing here is deployed anywhere —
there's no auth because there's no one on your own machine to protect it
from, and no cloud config because there's nowhere it's meant to run but
`localhost`. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design and
roadmap.

Everything runs locally and free: open-weight models, no paid APIs, no keys.

## Stack

Python · FastAPI · PyTorch/Hugging Face Transformers · Qdrant · PostgreSQL ·
React · TypeScript · Tailwind CSS · Docker

## Project layout

```
frontend/                  DONE — React + TypeScript + Tailwind, one folder, repo root
├── Dockerfile               multi-stage: node build -> nginx serve
├── nginx.conf               serves the app, proxies API paths, SSE-safe
└── src/
    ├── App.tsx
    ├── api/client.ts         fetch wrappers + the raw SSE stream reader
    ├── hooks/                useDocuments, useConversations, useStreamingQuery
    └── components/           ChatPanel, MessageBubble, CitationChip,
                               PagePreviewModal, ToolCallTrace, Sidebar, ...

eval/                       DONE — Phase 7 evaluation harness
├── questions.json
├── metrics.py                Recall@K, Precision@K, MRR
└── run_evaluation.py

src/copilot/
├── main.py             FastAPI app factory, CORS, exception handler
├── core/                settings (env-driven) + logging
├── api/
│   ├── routes/          health, documents, search, query, agent, conversations
│   └── sse.py            Server-Sent Events formatting
├── db/                   SQLAlchemy models (Document, Page, Chunk, Image,
│                          Conversation, Message) + session
├── schemas/              Pydantic request/response models
├── conversation/         DONE — Q&A history persistence (a log, not memory)
├── ingestion/            DONE — PDF -> text/tables/images -> chunks -> Postgres
│   ├── parser.py           pdfplumber (text + tables) + pypdf (images)
│   ├── chunker.py          page-scoped, structure-aware, overlapping chunks
│   ├── service.py          parse -> chunk -> persist orchestration
│   └── preview.py          render + cache a source page as PNG
├── retrieval/            DONE — text + image search, fused
│   ├── embedder.py         sentence-transformers, BGE query prefix
│   ├── vector_store.py     Qdrant collections, upsert, filtered search
│   ├── indexer.py          chunks -> vectors
│   ├── retriever.py        query -> Evidence with page numbers
│   ├── image_embedder.py   CLIP, shared text/image space
│   ├── image_indexer.py    images -> vectors (+ optional captions)
│   ├── image_retriever.py  CLIP search + page-context lookup
│   ├── captioner.py        optional VLM captions, off by default
│   └── multimodal.py       rank fusion over text and images
├── generation/           DONE — grounded answering with citations, streamed
│   ├── prompt.py           grounded prompt + citation parsing
│   ├── grounding.py        resolve citations against the evidence
│   ├── faithfulness.py     does the cited text actually support the claim?
│   ├── local_lm.py         shared model wrapper; chat() and chat_stream()
│   └── generator.py        local LLM (default) and VLM implementations
└── agent/                DONE — a tool-using agent, streamable
    ├── tools.py             search_documents, search_images, get_page,
    │                        calculate (safe AST evaluator), get_document_metadata
    ├── planner.py           single-shot LLM planning, with a Phase 5 fallback
    ├── orchestrator.py      run()/run_stream() a plan, reuses Phase 5's checks
    └── deps.py              lazy, cached agent assembly
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
      "fused_score": 0.0164,
      "text": "Overheating is most commonly caused by insufficient cooling airflow…"
    },
    {
      "kind": "image",
      "document_id": "0a1b…",
      "page_number": 37,
      "score": 0.221,
      "fused_score": 0.0161,
      "image_id": "9c2f…",
      "image_path": "data/images/0a1b…/page0037_img00.png"
    }
  ]
}
```

Results combine passages and diagrams. `score` is the similarity from whichever
model found the item — **comparable within a kind, not across kinds**, since
text comes from BGE and images from CLIP. `fused_score` is what actually
ordered the list; see [ARCHITECTURE.md](ARCHITECTURE.md) on rank fusion.

Options: `"document_id"` searches within one manual, `"include_images": false`
returns text only. Uploading indexes automatically; `POST
/documents/{id}/index` re-indexes (safe to repeat).

Failures are contained. If the text model or Qdrant is unavailable, uploads
still succeed and report `indexed_chunks: 0`. If only CLIP is unavailable, text
search is unaffected and `indexed_images` is 0 — you lose diagrams, not search.

## Asking a question

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "why is the pump overheating?", "top_k": 5}'
```

```json
{
  "answer": "Overheating is most commonly caused by insufficient cooling airflow across the motor fins [page 37]. A blocked intake filter reduces airflow and trips the thermal cutout [page 37]. This suggests checking the intake filter first.",
  "citations": [
    {"kind": "text",  "page_number": 37, "chunk_id": "8f1c…"},
    {"kind": "image", "page_number": 38, "image_id": "9c2f…", "image_path": "data/images/…/page0038_img00.png"}
  ],
  "insufficient_evidence": false,
  "unsupported_pages": [],
  "grounded": true,
  "faithfulness": 0.86,
  "tool_calls": [],
  "conversation_id": "1c9a…"
}
```

Pass `conversation_id` back on the next question to keep it in the same
history thread (see "Asking the agent" and ARCHITECTURE.md — this only
affects what gets logged, not how the question is answered). `POST
/query/stream` is the same endpoint delivered as Server-Sent Events instead
of one JSON blob — see "Using the app" for why that matters on a slow local
model.

Four fields carry the guarantees:

- **`citations`** are the pages the answer actually cited, resolved back to the
  retrieved evidence — not everything that was retrieved.
- **`insufficient_evidence`** is true when the model declined, or when nothing
  was retrieved at all. The answer text is not a claim in that case.
- **`unsupported_pages`** is non-empty when the answer cited a page that was
  never in the evidence. That means the model invented a source, and the answer
  should not be trusted. It is surfaced rather than hidden precisely because an
  invented citation *looks* authoritative.
- **`grounded`** is false when the answer makes a claim the retrieved evidence
  does not support — either it cites nothing, or it cites a real page whose
  text does not actually back the claim. The text may happen to be correct,
  but the manual does not back it, so it must not be shown as sourced.
- **`faithfulness`** is the share of the answer's content words found in the
  evidence it cited (0–1). `grounded` requires at least 0.5.

Model choice matters more than anything else here. Measured against real
Grundfos manuals, `Qwen2.5-0.5B-Instruct` emitted **no citations at all** and
answered off-topic questions instead of refusing. Upgrading to
`Qwen2.5-1.5B-Instruct` fixed that — and then, asked "What is the capital city
of France?", answered *"The capital city of France is Paris [page 21]"*: a
real, resolvable citation to a page that says nothing about France.
`unsupported_pages` was empty; the citation was genuine. Only `faithfulness`
catches this, which is why `grounded` checks both — does the citation resolve,
and does the cited text actually support the claim.

## Asking the agent

```bash
curl -X POST http://localhost:8000/agent/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Model A reaches 95C and model B tops out at 80C. What is the percentage difference?"}'
```

Same response shape as `/query`, plus `tool_calls` — a human-readable log of
what the agent actually did:

```json
{
  "answer": "The percentage difference is 18.75% [page 12].",
  "citations": [{"kind": "text", "page_number": 12, "chunk_id": "8f1c…"}],
  "insufficient_evidence": false,
  "unsupported_pages": [],
  "grounded": true,
  "faithfulness": 0.91,
  "tool_calls": [
    "search_documents(query='temperature limit specification') -> 3 result(s)",
    "calculate(expression='(95-80)/80*100') -> 18.75"
  ]
}
```

The agent decides for itself, from five tools, which ones a question needs:
`search_documents`, `search_images`, `get_page` (an exact page by number),
`calculate` (arithmetic, via a whitelisted-AST evaluator — no `eval`), and
`get_document_metadata` (which manuals exist, how many pages one has). A
calculation or a metadata lookup isn't sourced from any manual page, so it
reaches the model as a trusted "computed fact," not as citable evidence — the
`18.75` in the answer above is arithmetic the tool actually did, not the model
guessing, and the citation still resolves to a real retrieved page. `/query`
always reports `tool_calls: []`, since its fixed pipeline calls no tools —
useful for comparing the two side by side on the same question.

Planning is single-shot (the model writes its whole tool sequence up front,
not an iterative loop) and, when the model's JSON can't be parsed, falls back
to exactly `/query`'s own default behavior (search text + images) rather than
failing the request — see ARCHITECTURE.md for why.

`POST /agent/query/stream` streams the same pipeline: one `tool_calls` event
once planning and tool execution finish (seconds), then the answer token by
token, then the final `result` event with the same fields as above.

## Browsing history

```bash
curl http://localhost:8000/conversations                # every past exchange
curl http://localhost:8000/conversations/{id}            # one, with all messages
curl -X DELETE http://localhost:8000/conversations/{id}
```

Both `/query` and `/agent/query` (streaming or not) log every question and
its full answer here automatically. It's a log, not memory: pass the
`conversation_id` a response returned back on your next question to group it
in the same thread, but the question is still answered with zero context from
the turns before it — see ARCHITECTURE.md's "Conversation history is a log,
not memory."

## Evaluating

```bash
python -m eval.run_evaluation --manuals-dir path/to/your/pdfs
python -m eval.run_evaluation --manuals-dir path/to/your/pdfs --use-agent
```

`eval/questions.json` is a ground-truth set — gold pages for retrieval
questions, `expect_insufficient` for off-topic refusal-control questions,
`answer_must_contain` for a calculation question exercising the Phase 6
calculator. The harness ingests every PDF in `--manuals-dir` fresh, runs each
question through the real retrieval stack and (by default) the fixed Phase 5
pipeline, and reports Recall@K/Precision@K/MRR alongside the exact same
`grounded`/`faithfulness`/`insufficient_evidence` fields `/query` returns —
there is no separate scoring path, so the numbers describe the real API.
`--use-agent` runs the identical question set through `/agent/query`'s
pipeline instead, so the two are directly comparable. Results are written to
`eval/evaluation_results.json` (gitignored — a generated artifact, not
source) and printed as a summary table.

Scaling the question set to the project's target of 50–100 questions, or
pointing it at a different corpus, is a data change to `questions.json`, not
an architecture change.

Measured against three real Grundfos manuals: **Recall@5 0.97**, **MRR
0.82**, all 3 off-topic control questions correctly refused. Answered-question
faithfulness is a more mixed 0.39 — about a third of answered questions
omitted their `[page N]` citation entirely (a real, known 1.5B failure mode),
and some of the rest are likely the lexical faithfulness check being strict on
legitimate paraphrase rather than genuine fabrication. Full breakdown and
caveats in ARCHITECTURE.md.

## Tests

```bash
pytest                  # 338 unit tests, no model download
pytest -m integration   # 24 more, using the real embedding, CLIP and LLM models
```

The unit suite runs Qdrant in-process and swaps in deterministic stand-ins for
both embedders, so it needs neither a container nor model weights. The
integration suite loads the real models.

## Running locally

Requirements: Docker Desktop. That's it.

```bash
cp .env.example .env
docker compose up --build
```

This starts:

- `frontend` — the app: **http://localhost:3000**. nginx serves the built
  React UI and reverse-proxies every API call to `api` internally, so this is
  the only address you need.
- `api` — FastAPI directly, if you want it: http://localhost:8000 (docs at
  `/docs`)
- `postgres` — metadata store on port 5432 (exposed for local debugging only)
- `qdrant` — vector store on ports 6333/6334 (same)

Nothing here is deployed anywhere — every port above is bound to your own
machine, there's no authentication because there's no one else who can reach
it, and no cloud configuration exists in this repo. See ARCHITECTURE.md's
"Local-only, by design" section.

Check the API is alive:

```bash
curl http://localhost:8000/health
```

The first build downloads and bakes in the embedding model and installs
CPU-only PyTorch, so it takes a few minutes. Rebuilding after an edit to
`src/` currently redoes that (a known, documented tradeoff — see
ARCHITECTURE.md's Phase 8 section on why it's left as-is); rebuilding after
only editing `frontend/` is fast, since its Docker layer is independent.

## Frontend development without Docker

```bash
cd frontend
npm install
npm run dev
```

Opens on http://localhost:5173 with hot reload. Vite's dev server proxies
every API path to `http://localhost:8000` (see `vite.config.ts`), so run the
backend (`docker compose up postgres qdrant api`, or the bare-metal steps
below) alongside it. The frontend code always calls relative paths like
`/query`; whether those are resolved by Vite's dev proxy or by nginx in the
Docker build is the only difference between the two setups.

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

## Using the app

Upload a manual from the sidebar, then ask a question. **Direct** always
retrieves then answers (Phase 5's fixed pipeline); **Agent** decides for
itself which of five tools a question needs (Phase 6) — try a question that
needs arithmetic (e.g. "what's the percentage difference between 95°C and
80°C?") to see the calculator get used, or a page-number question to see
`get_page`. The answer streams in as it's generated — a real local answer
takes up to about a minute on CPU (see Phase 7's measured 73s mean generation
time), so this matters more than it might sound like it should.

Every answer carries **grounded**/**not grounded** and a **faithfulness**
score — click a citation chip to see the actual source page it came from.
Past conversations are listed in the sidebar and stay fully re-openable;
each question is still answered independently (no memory of prior turns —
see ARCHITECTURE.md).

## Roadmap

All 8 phases from the original plan are complete: ingestion, embeddings/vector
search, multimodal retrieval, grounded generation, the tool-using agent layer,
evaluation, and — this one — a full local application with a UI, streaming,
and history. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design of
each.
