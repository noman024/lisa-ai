# LISA — Life Insurance Support Assistant

- **Author:** MD Mutasim Billah Noman
- **Updated on:** 25 April 2026

A production-style AI chat agent that answers life insurance questions accurately using **retrieval-augmented generation (RAG)**. Built with **FastAPI**, **LangGraph**, **FAISS**, and **vLLM-Metal** (Apple Silicon) or any OpenAI-compatible LLM server.

### Skill test alignment (AI Agent: Life Insurance Support Assistant)


| Requirement                                 | How this repo satisfies it                                                                        |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Conversational text UI                      | `scripts/chat_cli.py` (also **OpenAPI** `/docs` + `POST /chat`)                                   |
| LLM (e.g. OpenAI API)                       | OpenAI-compatible client via `app/llm/client.py` — configure in `.env`                            |
| Context across turns                        | `session_id` + `app/memory/store.py` — prior turns in the graph prompt; facts from RAG only       |
| Policy types, benefits, eligibility, claims | `knowledge/insurance_kb.md` + router labels / retriever + grounded answers                        |
| Python 3.10+, LangGraph, FastAPI, storage   | `requirements.txt`, LangGraph in `app/agent/graph.py`, FAISS + JSON metadata + in-memory sessions |
| **Bonus:** LangGraph & configurable KB      | Graph pipeline; `KNOWLEDGE_PATH` / `DATA_DIR` in `.env.example` and `app/config.py`               |


## Architecture

```
User / CLI
    │
    ▼  POST /chat
FastAPI  ──────────────────────────────────────────┐
    │                                              │
    ▼  LangGraph pipeline                          │
 router → retriever → prompt_builder → llm → validator
              │                          │
              ▼                          ▼
         FAISS + BGE               vLLM / OpenAI
         (local)                   (port 8001)
```

**Pipeline steps:**


| Node             | What it does                                                                                                                                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `router`         | Classifies query as `informational / eligibility / claims / comparison`, or `off_topic` for obvious non-insurance topics (regex). Flags `vague_query` for very short or underspecified text (stricter retrieval threshold).           |
| `retriever`      | Dense vector search over the knowledge base (BGE embeddings + FAISS inner product). Queries longer than 4000 characters are truncated **only for the embedding step**; the full question text is still sent to the LLM in the prompt. |
| `prompt_builder` | Builds the user message: optional `Recent conversation` (prior turns for this `session_id`) + `Question` + top-3 chunks as `Context` + `Answer:`                                                                                      |
| `llm`            | Calls the configured OpenAI-compatible chat endpoint                                                                                                                                                                                  |
| `validator`      | Checks grounding — fraction of answer content words that appear in the context                                                                                                                                                        |


**Guardrails:** If the best retrieval score is below `RETRIEVAL_MIN_SCORE` (scaled up when `vague_query` is true and there is prior conversation), the router marks the turn as `off_topic`, or the answer's grounding score is below `GROUNDING_MIN_OVERLAP`, the pipeline returns `"I don't know."` instead of a hallucinated answer. Off-topic turns skip embedding search entirely. **Standalone vague** messages (no prior turns) skip retrieval entirely so short replies like “ok” cannot match random chunks.

## Project Layout

```
lisa-ai/
├── app/
│   ├── main.py          # FastAPI app factory and lifespan
│   ├── config.py        # Settings (pydantic-settings, .env)
│   ├── api/routes.py    # GET /, /health, POST /chat
│   ├── agent/
│   │   ├── context.py   # AgentContext dataclass (wired at startup)
│   │   ├── graph.py     # LangGraph compile
│   │   ├── nodes.py     # All five node implementations
│   │   └── state.py     # GraphState TypedDict
│   ├── llm/client.py    # AsyncOpenAI wrapper
│   ├── memory/store.py  # In-process session history
│   ├── models/schemas.py  # Pydantic request/response models
│   ├── rag/
│   │   ├── embeddings.py  # BGE sentence-transformers (singleton)
│   │   └── retriever.py # FAISS flat-IP retriever
│   └── utils/
│       ├── chunking.py  # Token-aware markdown chunker (tiktoken)
│       └── grounding.py # Answer-vs-context word-overlap score
├── tests/               # pytest suite (API, graph, RAG, utils)
├── knowledge/
│   └── insurance_kb.md  # Life insurance knowledge base (editable)
├── scripts/
│   ├── ingest_kb.py     # Build FAISS index from knowledge base
│   ├── start_vllm.sh    # Start vLLM-Metal server (Apple Silicon)
│   ├── dev_stack.sh     # Ingest KB + start LISA API
│   ├── start_all.sh     # One-command: vLLM + ingest + LISA API
│   ├── chat_cli.py      # Interactive terminal chat client
│   ├── verify_e2e.py    # Structured end-to-end health check
│   └── eval_queries.py  # Smoke-test a fixed set of queries
├── data/                # Auto-generated (gitignored)
│   ├── faiss.index
│   └── metadata.json
├── .env.example
├── pytest.ini           # pytest config (pythonpath, asyncio)
└── requirements.txt
```

## Prerequisites


| Requirement                         | Details                                                                                               |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Python 3.10+** (3.10–3.14 tested) | App venv: `python3 -m venv .venv` — use a version that matches your stack.                            |
| **Python 3.12**                     | Required by the `vllm-metal` venv (installed separately)                                              |
| **macOS on Apple Silicon**          | For `vllm-metal`; see [OpenAI and other remote LLMs](https://github.com/vllm-project/vllm-metal.git) for other systems |
| **~3 GB free RAM**                  | Qwen2.5-3B-Instruct-4bit model (downloaded once to `~/.cache/huggingface`)                            |


## Setup

### 1 — Clone and create the LISA venv

```bash
git clone https://github.com/noman024/lisa-ai.git && cd lisa-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # edit if needed; defaults work out of the box
```

### 2 — Install vLLM-Metal (Apple Silicon, one-time)

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

This installs `vllm` + the `vllm-metal` MLX plugin into `~/.venv-vllm-metal` using Python 3.12. The default model (`mlx-community/Qwen2.5-3B-Instruct-4bit`, ~2 GB) is downloaded automatically on first server start.

### 3 — Build the knowledge base index

```bash
export PYTHONPATH=.
python scripts/ingest_kb.py
# → Wrote 8 chunks to data/faiss.index and data/metadata.json
```

Chunks `knowledge/insurance_kb.md` into ~400-token windows with overlap, encodes with `BAAI/bge-small-en-v1.5`, and writes the FAISS index. Re-run whenever you edit the knowledge file.

## Running the Full Stack

### Option A — Two terminals (recommended for development)

**Terminal A** — LLM server (keep running):

```bash
bash scripts/start_vllm.sh
# Serves mlx-community/Qwen2.5-3B-Instruct-4bit on http://127.0.0.1:8001/v1
```

**Terminal B** — LISA API (after vLLM is ready):

```bash
bash scripts/dev_stack.sh
# Ingests KB (idempotent) then serves LISA on http://0.0.0.0:8000
```

### Option B — Single command (background processes)

```bash
bash scripts/start_all.sh
# Starts vLLM, waits for it to be ready, ingests KB, then starts LISA.
# Ctrl-C stops all services cleanly.
```

The wait loop probes `GET /v1/models` with `Authorization: Bearer` set to the same value as vLLM’s `--api-key` (default `not-needed` via `VLLM_API_KEY`). A bare `curl` to `/v1/models` returns 401, so the script must send that header or readiness never succeeds.

## API Reference

### `GET /`

```bash
curl -s http://127.0.0.1:8000/
```

JSON with `service` name, `version`, and pointers to `docs`, `health`, and `chat` (for a quick check in a browser or before recording a demo).

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "llm_base_url": "http://127.0.0.1:8001/v1",
  "llm_model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
  "embedding_model_id": "BAAI/bge-small-en-v1.5",
  "index_ready": true,
  "index_error": null
}
```

When the FAISS index is missing or invalid, `index_ready` is `false` and `index_error` is a short reason (e.g. run `scripts/ingest_kb.py`).

### `POST /chat`

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "What is term life insurance?"}'
```

```json
{
  "response": "Term life insurance provides a death benefit to beneficiaries if the insured dies during a specified term (10, 20, or 30 years). It does not build cash value.",
  "sources": ["Term Life Insurance", "Comparison: Term vs Whole Life (Summary)"],
  "query_type": "informational",
  "low_confidence": false
}
```

**Request fields:**


| Field        | Type   | Description                                                                                                                  |
| ------------ | ------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `session_id` | string | Identifies the conversation; prior turns are stored server-side and included in the LLM prompt (not used as factual context) |
| `message`    | string | User's question (max 8000 chars; leading/trailing whitespace stripped; all-whitespace rejected)                              |


**Response fields:**


| Field            | Type     | Description                                                                                                                   |
| ---------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `response`       | string   | Answer grounded in the knowledge base, or `"I don't know."`                                                                   |
| `sources`        | string[] | KB sections the answer drew from                                                                                              |
| `query_type`     | string   | `informational / eligibility / claims / comparison / off_topic`                                                               |
| `low_confidence` | bool     | `true` if retrieval was weak, the question was off-topic/vague, the LLM call failed, or the answer failed the grounding check |


## CLI Chat

```bash
export PYTHONPATH=.
python scripts/chat_cli.py
# or: python scripts/chat_cli.py --base-url http://127.0.0.1:8000

# Commands:  help  |  clear (new session)  |  quit / q
# After connect, the CLI shows LLM, embedding model, and index status from /health.
```

## Automated tests (pytest)

Runs without a local vLLM or a built FAISS index (startup is mocked). The project `pytest.ini` sets `pythonpath = .` so you can run `pytest` from the repo root without `export PYTHONPATH=.` (exporting it is still fine).

```bash
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Optional coverage report for `app/`:

```bash
pytest -q --cov=app --cov-report=term-missing:skip-covered
```

Tests cover API validation (empty / whitespace messages, max length 8000 and overflow, malformed JSON, session id length), multi-turn `history` and session store behaviour, LangGraph pipeline with mocked LLM/retriever, embedding client with `SentenceTransformer` mocked, chunking and grounding utilities, FAISS empty-query and long-query truncation, index load/dimension mismatch, optional `LLM_SEED` wiring, LLM error fallbacks (`OpenAIError` and other failures), router/retriever edge cases (off-topic, vague stricter threshold, short non-English text), validator/refusal paths, and HTTP responses when the graph reports `low_confidence` or `off_topic`.

## End-to-End Verification

```bash
export PYTHONPATH=.
python scripts/verify_e2e.py --llm-api-key not-needed
# Checks: GET /health → GET /v1/models → POST /chat (validates JSON shape + HTTP 200)
```

```bash
# Skip the LLM server ping (e.g. CI without a running vLLM):
python scripts/verify_e2e.py --skip-llm-ping
```

## Sample Conversation

```
You: What types of life insurance policies exist?
Lisa: The main types are Whole Life Insurance and Term Life Insurance.
      Whole Life provides lifetime coverage with fixed premiums and a cash
      value component. Term Life covers a specified period (10–30 years)
      without cash value. Universal Life adds flexible premiums and an
      adjustable death benefit.

You: Who qualifies for life insurance?
Lisa: Eligibility depends on age, health (medical history, labs, build),
      lifestyle (hazardous activities may be rated or excluded), residency,
      financial justification, and legal capacity to contract.

You: How do I file a claim?
Lisa: 1. Notify the insurer as soon as possible.
      2. Obtain claim forms from the carrier or agent.
      3. Submit a certified death certificate, completed claim form, and ID.
      4. The insurer verifies policy status and cause of death.
      5. Payout options include lump sum, annuity, or retained asset account.

You: What about my Alaska-specific policy limit?
Lisa: I don't know.
      (State-specific rules are not in the knowledge base.)
```

## Configuration

All settings are in `.env` (copy from `.env.example`). Key variables:


| Variable                     | Default                                  | Description                                                               |
| ---------------------------- | ---------------------------------------- | ------------------------------------------------------------------------- |
| `DATA_DIR` / `LISA_DATA_DIR` | Project `./data`                         | FAISS index and `metadata.json` location                                  |
| `KNOWLEDGE_PATH`             | `knowledge/insurance_kb.md`              | RAG source markdown; re-run `ingest_kb.py` after changes                  |
| `VLLM_BASE_URL`              | `http://127.0.0.1:8001/v1`               | LLM server base URL                                                       |
| `VLLM_MODEL`                 | `mlx-community/Qwen2.5-3B-Instruct-4bit` | Served model name                                                         |
| `VLLM_API_KEY`               | `not-needed`                             | API key for vLLM (any string)                                             |
| `EMBEDDING_MODEL_ID`         | `BAAI/bge-small-en-v1.5`                 | HuggingFace sentence-transformers model                                   |
| `RETRIEVAL_MIN_SCORE`        | `0.32`                                   | Minimum similarity (IP on normalized BGE) to trust retrieval              |
| `GROUNDING_MIN_OVERLAP`      | `0.25`                                   | Minimum word overlap (answer vs context)                                  |
| `RETRIEVER_TOP_K`            | `5`                                      | Number of chunks to retrieve                                              |
| `MEMORY_MAX_MESSAGES`        | `20`                                     | Max user+assistant lines stored; caps prompt history lines                |
| `MEMORY_PROMPT_MAX_CHARS`    | `4000`                                   | Max prior-conversation text in the LLM user prompt (tail)                 |
| `LLM_SEED` / `VLLM_SEED`     | (unset)                                  | Optional: passed to the chat API when supported (more deterministic runs) |
| `LLM_TIMEOUT_SECONDS`        | `120`                                    | HTTP timeout for LLM calls                                                |
| `LLM_MAX_TOKENS`             | `600`                                    | Max tokens to generate                                                    |
| `LLM_TEMPERATURE`            | `0.1`                                    | Sampling temperature                                                      |


The config supports alias groups so `VLLM_*`, `LLM_*`, and `OPENAI_*` all map to the same fields.

## OpenAI and other remote LLMs

Use any OpenAI-compatible API (including OpenAI’s hosted API) so you do **not** need a local vLLM server. Uncomment the block in `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

Then run only the LISA API (no vLLM needed):

```bash
bash scripts/dev_stack.sh
```

## Extending the Knowledge Base

Edit `knowledge/insurance_kb.md` (markdown with `##` section headers), then re-ingest:

```bash
export PYTHONPATH=.
python scripts/ingest_kb.py
```

Restart the LISA API — it loads `data/faiss.index` at startup.

## Dependencies


| Package                 | Purpose                                         |
| ----------------------- | ----------------------------------------------- |
| `fastapi` + `uvicorn`   | HTTP API server                                 |
| `langgraph`             | Agent graph execution                           |
| `openai`                | OpenAI-compatible HTTP client (works with vLLM) |
| `sentence-transformers` | Local BGE embeddings                            |
| `faiss-cpu`             | Vector index and similarity search              |
| `tiktoken`              | Token counting for chunking                     |
| `pydantic-settings`     | Typed configuration from `.env`                 |


**LLM server** (installed separately, not in `requirements.txt`):

```bash
# Apple Silicon — installs to ~/.venv-vllm-metal
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

## License

This project is a technical demonstration. The `knowledge/insurance_kb.md` file is for educational testing only — not legal, tax, or investment advice. Consult a licensed professional for personal insurance decisions.