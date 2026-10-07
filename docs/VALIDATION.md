# Validation report

Checked on October 7, 2026, using Python 3.11 on macOS.

## Observed results

| Check | Result |
| --- | --- |
| `pytest -q` | **32 passed** |
| `ruff check app tests` | Passed |
| `ruff format --check app tests` | Passed, 15 files formatted |
| `mypy app` | Passed, no issues in 10 source files |
| `python -m pip check` | Passed, no broken requirements |
| Python compilation of app and live demo | Passed |

One third-party deprecation warning is emitted: the installed Starlette test client plans to move from `httpx` to `httpx2`. It does not fail tests. This project currently uses the tested, pinned `httpx` dependency.

## What was exercised

- Both chunking strategies, overlap boundaries, oversized-paragraph fallback and invalid settings.
- Valid text-based PDF extraction and API ingestion; TXT ingestion and SQL/Qdrant persistence.
- Malformed, empty, binary, encrypted and unsupported inputs; upload limits.
- Failed embedding and partial vector-write cleanup.
- Authentication and the two-endpoint API schema.
- Invalid request fields and unknown document IDs.
- RAG source filtering by document, follow-up rewriting, session isolation and Redis TTL.
- Empty retrieval and invalid model citation indices.
- Multi-turn booking, missing fields, malformed email, impossible/past dates and corrections.
- Confirmation before persistence, exact start-time collision and draft cancellation.
- Duplicate request replay, conflicting reuse, bounded history and replay without rewinding memory.
- Per-session lock contention.
- Redis failure after SQL commit followed by safe replay.
- SQL failure during chat persistence rolling back the booking as well.
- Actual OpenAI Python SDK serialization and parsing using mocked HTTP transport for embeddings and structured Responses calls.

SQLite and Qdrant's local engine are real implementations in tests. Redis is `fakeredis` with Lua support. Model results are mocked; no real LLM or embedding API was called. This distinction matters: the tests cannot measure semantic retrieval quality, prompt effectiveness or real model extraction accuracy.

## Not verified in this environment

- Docker build and the running multi-container stack: Docker was not available.
- Redis server network behavior and remote Qdrant server connectivity.
- Live OpenAI credentials, model access, quota, generation quality or latency.
- Linux CI execution: the workflow is supplied but has not run on GitHub here.
- GitHub publication and CI status are tracked by the repository commit and Actions run; they are not part of the local test result.

## Before submission

1. Set real secrets in `.env`, then run `docker compose up --build -d`.
2. Confirm the API starts with the Qdrant and Redis services reachable.
3. Run `python examples/demo.py` with `API_KEY` set. This makes provider calls and saves a local demo booking.
4. Read the displayed answers: check that claims are supported by the printed source text and the follow-up resolves the intended subject.
5. Try a new booking session with a malformed email, past date, correction and explicit confirmation.
6. Confirm a request replay returns the same booking ID.
7. Push to your repository and check the CI run before sending HR its link.

Do not describe live integration or semantic quality as verified until these live steps are completed.
