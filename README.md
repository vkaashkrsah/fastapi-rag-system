# fastapi-rag-system

PalmMind AI/ML internship assignment: a modular conversational RAG backend.

A backend with **two business REST APIs**: upload a document, then ask questions or book an interview through a conversation. There is no frontend and no LangChain dependency.

## Requirements covered

| HR requirement | Implementation |
| --- | --- |
| FastAPI or similar | FastAPI with typed requests and responses |
| Upload `.pdf` and `.txt` | `POST /api/v1/documents`; pypdf and UTF-8 decoding |
| Two selectable chunking strategies | `fixed` with overlap; `paragraph` with bounded fallback |
| Embeddings and permitted vector database | OpenAI embeddings stored in Qdrant |
| SQL/NoSQL metadata | SQLite through SQLAlchemy: document and chunk metadata |
| Custom RAG | Explicit rewrite → embed → retrieve → grounded generation |
| Redis chat memory | Session history and booking draft; TTL and bounded messages |
| Multi-turn queries | LLM rewrites follow-up questions using Redis history |
| LLM interview booking | Structured extraction of name, email, date, time across turns |
| Persist bookings | Validated bookings committed to SQLite |
| No FAISS, Chroma, RetrievalQAChain or UI | None included; interactive documentation UI is disabled |
| Clean modular code and typing | Separated modules, Pydantic models, Protocol, strict mypy |

## Start with Docker

Prerequisites: Docker with Compose and an OpenAI API key with access to the configured models. The default models are `gpt-4o-mini` and `text-embedding-3-small`; they are configurable and are not claimed to be the newest models. API calls use your provider account and may incur charges.

Clone this repository and configure the environment:

```sh
git clone https://github.com/vkaashkrsah/fastapi-rag-system.git
cd fastapi-rag-system
cp .env.example .env
```

Edit `.env` and set:

- `OPENAI_API_KEY`: your real provider key.
- `API_KEY`: a random secret for protecting these two endpoints. You can generate one using `python3 -c 'import secrets; print(secrets.token_urlsafe(32))'`.
- `BOOKING_TIMEZONE`: the timezone for date/time interpretation, default `Asia/Kolkata`.

Then:

```sh
docker compose up --build -d
docker compose logs -f api
```

Wait for `Application startup complete`. Compose starts Qdrant and Redis on its internal network. The API retries by container restart if Qdrant is still starting. Only `127.0.0.1:8000` is exposed to your computer. SQLite, Qdrant, and Redis use persistent named volumes.

API schema: `http://localhost:8000/openapi.json`. This is machine-readable JSON, not a UI. There are exactly two business endpoints.

Stop without deleting data:

```sh
docker compose down
```

## API 1 — Document ingestion

Set the API secret in your terminal (use the same value as `.env`):

```sh
export API_KEY='your-api-secret'
```

Upload with fixed-size chunking:

```sh
curl --fail-with-body http://localhost:8000/api/v1/documents \
  -H "Authorization: Bearer $API_KEY" \
  -F 'file=@examples/handbook.txt' \
  -F 'strategy=fixed' -F 'chunk_size=1000' -F 'overlap=100'
```

Upload with paragraph chunking:

```sh
curl --fail-with-body http://localhost:8000/api/v1/documents \
  -H "Authorization: Bearer $API_KEY" \
  -F 'file=@examples/handbook.pdf' \
  -F 'strategy=paragraph' -F 'chunk_size=1000' -F 'overlap=0'
```

Example response (IDs and counts vary):

```json
{
  "document_id": "fd60172c-18fa-40cb-bfb1-69005624d22e",
  "filename": "handbook.txt",
  "strategy": "fixed",
  "chunk_count": 1,
  "status": "ready"
}
```

Save `document_id`; supply it to every document-question request.

| Form field | Meaning |
| --- | --- |
| `file` | One UTF-8 TXT or text-based PDF |
| `strategy` | `fixed` (default) or `paragraph` |
| `chunk_size` | Characters per chunk, 100–2000; default 1000 |
| `overlap` | Fixed defaults to 100; paragraph defaults to 0 and requires 0 |

For fixed chunks, `overlap` must be smaller than `chunk_size`; explicitly set a smaller overlap if choosing `chunk_size=100`. Paragraphs are packed without breaking them when possible. A paragraph longer than the limit is split into bounded character windows.

Default limits: 10 MiB upload, 200 PDF pages, 500,000 extracted characters, 1,000 chunks. Empty, corrupted, encrypted, image-only PDFs and invalid UTF-8 files are rejected with actionable errors. OCR is outside the assignment scope. Uploaded originals are not saved; text chunks are saved in Qdrant and metadata in SQLite.

## API 2 — Conversation and booking

Use a UUID `session_id` for one conversation and a fresh UUID `request_id` for each new message. Reuse **the same request ID and identical payload** when retrying a failed request. Repeating a successful request returns its stored result, without another LLM call or booking.

```sh
curl --fail-with-body http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "4b067480-551e-4b2c-ac5c-33093ce2c44e",
    "request_id": "15cb7076-39d3-4b9d-9bb7-d18bb42e2532",
    "message": "What skills are useful for the internship?",
    "document_ids": ["REPLACE-WITH-THE-UPLOADED-DOCUMENT-UUID"]
  }'
```

The placeholder in `document_ids` must be replaced with a real returned UUID before running this command. Responses contain `answer`, `sources`, and optional `booking` / `booking_draft`. Each source includes the document ID, filename, chunk index, similarity score, and supporting text. Sources are taken from actual Qdrant hits, not model-invented references.

Follow-up: send `"Tell me more about those skills"` with the same session and document IDs, but a new request ID. Redis history lets the LLM expand “those skills” into a standalone search question.

Booking uses this same endpoint. `document_ids` may be empty. Example conversation:

1. `Book an interview. My name is Vikash and my email is vikash@example.com.`
2. The backend asks for date and time.
3. `2099-10-10 at 14:30` (use your actual desired future date).
4. The backend displays the four fields and configured timezone and asks for confirmation.
5. `confirm`
6. The backend saves the booking and returns its ID.

Corrections require fresh confirmation. Invalid email, impossible/past date, invalid time, and occupied exact start time are rejected. The example assumes one interviewer and unique start times; it does not model interview duration, business hours, or overlapping time intervals. The booking is a database record; no calendar invitation or email is sent, as neither is required by HR.

An explicit cancellation clears only an unfinished draft. Rescheduling/deleting saved bookings is outside these two API requirements.

### End-to-end live demo

The included command-line script uploads both file types, asks two related questions, then makes a test booking. It uses live provider calls and saves a real record in your local assignment database:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.lock
python examples/demo.py
```

Set `API_KEY` first. Set `BASE_URL` only if it differs from `http://localhost:8000`. The script prints the responses so you can show them to a reviewer. It is a command-line demo, not a frontend.

## Run and verify locally

Docker is the simplest way to run all services. To run only the Python backend on your host, first provide reachable Redis and Qdrant services at the URLs in `.env`, then:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.lock
pip install --no-deps -e .
uvicorn app.main:app --reload
```

Automated tests do **not** require Docker or API keys:

```sh
pytest -q
ruff check app tests
ruff format --check app tests
mypy app
```

Tests exercise real FastAPI routes, a temporary SQLite database, Qdrant's local implementation, and a Redis-compatible test double with Lua lock support. LLM outputs are deterministic test doubles. A separate adapter test exercises the actual OpenAI SDK against mocked HTTP responses. These tests validate orchestration and SDK contracts, not live model reasoning quality or remote service connectivity. See [validation report](docs/VALIDATION.md) for observed results and the live checklist.

`requirements.lock` pins runtime dependencies; `requirements-dev.lock` pins the tested Python 3.11 development environment. CI repeats lint, formatting, typing, and tests. Pin files are resolved versions, not cryptographic package hash locks.

## Code map

```text
app/
  main.py          # FastAPI routes, authentication, startup, dependency wiring
  config.py        # Environment settings and validation
  schemas.py       # API contracts, conversation state, structured model outputs
  documents.py     # PDF/TXT extraction and the two chunkers
  llm.py           # Embeddings, conversation planning, grounded answer calls
  vector_store.py  # Qdrant creation, storage, filtered retrieval, cleanup
  memory.py        # Redis state, TTL and session lock
  database.py      # SQLAlchemy document/chunk/booking/replay tables
  services.py      # Ingestion, RAG and booking orchestration
```

Read [the beginner walkthrough](docs/EXPLAIN_TO_HR.md) for simple explanations, implementation steps, interview answers and flowcharts. See [architecture](docs/ARCHITECTURE.md) for transaction and failure details.

## Scope and limitations

This is a runnable assignment backend, not a claim of production deployment readiness.

- A shared bearer key protects a single trusted workspace. There are no per-user accounts or tenant boundaries; document UUIDs are search filters, not access-control credentials.
- Redis memory expires after 24 hours of inactivity by default and keeps 20 messages. SQL replay snapshots remain durable; they are used when replaying requests, not as the normal conversational memory. Long conversations lose older context. Establish a retention policy before using real personal data.
- Booking intent, extraction and query rewriting use an LLM. Structured output constrains shape, not factual correctness. Validation and confirmation reduce mistakes; prompts do not guarantee resistance to all prompt injection.
- Grounding prompts, source validation and a similarity threshold reduce unsupported answers, but they cannot prove every claim is entailed. Evaluate retrieval and answers on real sample documents before deployment.
- Qdrant and SQL do not share a transaction. Documents become searchable only after SQL marks them ready. Failures trigger best-effort vector deletion. A process crash can leave pending documents/orphaned vectors requiring reconciliation.
- Re-uploading a file creates a new document; ingestion is not deduplicated. SHA-256 is stored as metadata.
- Embedding model/dimension changes require a new collection and re-ingestion. Do not mix embeddings from different models in one collection.
- Requests are synchronous and run in FastAPI's thread pool. This is suitable for a bounded assignment demo; large ingestion workloads should use a task queue. Session locking has a five-minute lease, not an indefinite distributed guarantee.
- Multipart parsing occurs before the route reads the file. Use an ingress request-body limit and resource limits if exposing it publicly; route checks alone do not protect against every oversized/malicious PDF.
- Production additions would include per-user authorization, rate limits, TLS, database migrations, backups, retention/deletion controls, monitoring and larger-scale concurrency testing.

## Submission

Repository: [vkaashkrsah/fastapi-rag-system](https://github.com/vkaashkrsah/fastapi-rag-system).

Run the live demo with your own environment, check the GitHub Actions result, and share this repository link with HR. Never commit `.env` or database files. Do not claim live validation unless you have actually run it. The explanation guide is intended for understanding and interview preparation; verify the code and explain any assistance honestly.

## Official implementation references

- [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/)
- [Qdrant local quickstart and query API](https://qdrant.tech/documentation/quickstart/)
- [Redis Python client](https://redis.io/docs/latest/integrate/redis-py/)
- [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings)
- [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
