# LISA — Life Insurance Support Assistant

A production-style AI chat agent that answers life insurance questions accurately using **retrieval-augmented generation (RAG)**. Built with **FastAPI**, **LangGraph**, **FAISS**, and **vLLM-Metal** (Apple Silicon) or any OpenAI-compatible LLM server.

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


| Node             | What it does                                                                                       |
| ---------------- | -------------------------------------------------------------------------------------------------- |
| `router`         | Classifies query as `informational / eligibility / claims / comparison` (regex, no extra LLM call) |
| `retriever`      | Dense vector search over the knowledge base (BGE embeddings + FAISS inner product)                 |
| `prompt_builder` | Constructs `Question → Context → Answer:` prompt from top-3 retrieved chunks                       |
| `llm`            | Calls the configured OpenAI-compatible chat endpoint                                               |
| `validator`      | Checks grounding — fraction of answer content words that appear in the context                     |


**Guardrails:** If the best retrieval score is below `RETRIEVAL_MIN_SCORE`, or the answer's grounding score is below `GROUNDING_MIN_OVERLAP`, the pipeline returns `"I don't know."` instead of a hallucinated answer.

## Project Layout

```
lisa-ai/
├── app/
│   ├── main.py          # FastAPI app factory and lifespan
│   ├── config.py        # Settings (pydantic-settings, .env)
│   ├── api/routes.py    # GET /health, POST /chat
│   ├── agent/
│   │   ├── context.py   # AgentContext dataclass (wired at startup)
│   │   ├── graph.py     # LangGraph compile
│   │   ├── nodes.py     # All five node implementations
│   │   └── state.py     # GraphState TypedDict
│   ├── llm/client.py    # AsyncOpenAI wrapper
│   ├── memory/store.py  # In-process session history
│   ├── models/schemas.py# Pydantic request/response models
│   ├── rag/
│   │   ├── embeddings.py# BGE sentence-transformers (singleton)
│   │   └── retriever.py # FAISS flat-IP retriever
│   └── utils/
│       ├── chunking.py  # Token-aware markdown chunker (tiktoken)
│       └── grounding.py # Answer-vs-context word-overlap score
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
└── requirements.txt
```

## Prerequisites


| Requirement                | Details                                                                                       |
| -------------------------- | --------------------------------------------------------------------------------------------- |
| **Python 3.10–3.12**       | The LISA app venv (`python3 -m venv .venv`)                                                   |
| **Python 3.12**            | Required by the `vllm-metal` venv (installed separately)                                      |
| **macOS on Apple Silicon** | For `vllm-metal`; see [OpenAI alternative](#option-openai-api-no-local-gpu) for other systems |
| **~3 GB free RAM**         | Qwen2.5-3B-Instruct-4bit model (downloaded once to `~/.cache/huggingface`)                    |


## Setup

### 1 — Clone and create the LISA venv

```bash
git clone <repo-url> && cd lisa-ai
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
  "index_ready": true
}
```

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


| Field        | Type   | Description                                                |
| ------------ | ------ | ---------------------------------------------------------- |
| `session_id` | string | Identifies the conversation; history is stored server-side |
| `message`    | string | User's question (max 8000 chars)                           |


**Response fields:**


| Field            | Type     | Description                                                 |
| ---------------- | -------- | ----------------------------------------------------------- |
| `response`       | string   | Answer grounded in the knowledge base, or `"I don't know."` |
| `sources`        | string[] | KB sections the answer drew from                            |
| `query_type`     | string   | `informational / eligibility / claims / comparison`         |
| `low_confidence` | bool     | `true` if retrieval score was below threshold               |


## CLI Chat

```bash
export PYTHONPATH=.
python scripts/chat_cli.py
# or: python scripts/chat_cli.py --base-url http://127.0.0.1:8000

# Commands inside the session:
#   quit / exit / q  — end
#   clear            — start a new session_id
```

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


| Variable                | Default                                  | Description                                  |
| ----------------------- | ---------------------------------------- | -------------------------------------------- |
| `VLLM_BASE_URL`         | `http://127.0.0.1:8001/v1`               | LLM server base URL                          |
| `VLLM_MODEL`            | `mlx-community/Qwen2.5-3B-Instruct-4bit` | Served model name                            |
| `VLLM_API_KEY`          | `not-needed`                             | API key for vLLM (any string)                |
| `EMBEDDING_MODEL_ID`    | `BAAI/bge-small-en-v1.5`                 | HuggingFace sentence-transformers model      |
| `RETRIEVAL_MIN_SCORE`   | `0.25`                                   | Minimum cosine similarity to trust retrieval |
| `GROUNDING_MIN_OVERLAP` | `0.15`                                   | Minimum word overlap (answer vs context)     |
| `RETRIEVER_TOP_K`       | `5`                                      | Number of chunks to retrieve                 |
| `MEMORY_MAX_MESSAGES`   | `20`                                     | Max messages stored per session              |
| `LLM_TIMEOUT_SECONDS`   | `180`                                    | HTTP timeout for LLM calls                   |
| `LLM_MAX_TOKENS`        | `512`                                    | Max tokens to generate                       |
| `LLM_TEMPERATURE`       | `0.1`                                    | Sampling temperature                         |


The config supports alias groups so `VLLM_*`, `LLM_*`, and `OPENAI_*` all map to the same fields.

## Option: OpenAI API (no local GPU)

Uncomment the OpenAI block in `.env`:

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