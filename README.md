
# ExperimentIQ

ExperimentIQ is a standalone query-routing RAG assistant for MLflow experiment tracking. It answers natural-language questions over MLflow runs, flow configs, logs, and the dataset registry by combining:
- **Structured retrieval** via MLflow Tracking API / text-to-SQL
- **Unstructured retrieval** via vector search (Qdrant + embeddings)
- **Route routing** to decide which path a question takes

## Architecture

```
User Question
  -> Router (rule-based)
       -> Structured path: MLflow Search API
       -> Unstructured path: Qdrant Vector DB
       -> Answer Builder (LLM + sources)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your MLflow tracking URI and LLM settings
```

### 3. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Query the API

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"query": "which flow config gave the best validation NLL?"}'
```

### 5. Run evaluations

```bash
python -m app.eval.benchmark.run --output results.json
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | POST | Ask a question, returns answer + sources + route |
| `/health` | GET | Health check |
| `/status` | GET | Detailed service status |

## Evaluation

Measured locally on 2026-10-14 (MLflow @ 127.0.0.1:5000, Qdrant local, `scripts/run_eval.py`, 10 questions):

| Metric | Score |
|---|---|
| Routing accuracy | **10/10 (100%)** |
| Answer accuracy | **10/10 (100%)** |
| Hallucination rate | **0/10 (0%)** |

Per-category: 6 structured (metric_search / metric_range / comparison) all correct with run IDs, configs, and metric values cited; 4 unstructured (log_retrieval / config_qa) all grounded in seeded logs/configs/notes.

Reproduce:

```bash
python scripts/seed_docs.py
python scripts/run_eval.py
```

> The benchmark contains hand-authored questions across:
- **metric_comparison**: "which run had the highest accuracy?"
- **metric_filter**: "show me runs with val_nll < 0.5"
- **log_retrieval**: "what do the training logs say?"
- **config_query**: "explain flow configuration"

## Tech Stack

- Python 3.11+
- FastAPI + Uvicorn
- MLflow Tracking API
- Qdrant (vector database)
- sentence-transformers (embeddings)

# ExperimentIQ

**ExperimentIQ** — Query-routing RAG assistant for MLflow experiment tracking.

A standalone RAG system that answers natural-language questions over MLflow runs, flow configs, logs, and the dataset registry. Unlike a basic "chat over PDFs" demo, ExperimentIQ splits retrieval by data modality:

- **Structured queries** (run metrics, params, config fields, dataset registry) route to **MLflow's search API / text-to-SQL**
- **Unstructured content** (logs, config text, notes) routes to **hybrid dense/BM25 vector retrieval**
- A **router** decides which path a question takes based on its semantics and key terms

Built in Python with FastAPI, MLflow Tracking API, Qdrant vector database, and any LLM provider via a thin client layer.

---

## Architecture

```
┌──────────────┐
│   User Query  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────┐
│      Router (router.py)      │
│  Rule-based + optional LLM   │
│  decides: structured vs      │
│  unstructured retrieval      │
└──────┬──────────────────────┘
       │
    ┌──┴──┐
    ▼     ▼
Structured  Unstructured
┌──────────┐  ┌───────────────┐
│ MLflow   │  │ Embedding +   │
│ Search   │  │ Qdrant + BM25 │
│ API      │  │ (hybrid)      │
└──────────┘  └───────────────┘
    │                 │
    ▼                 ▼
  Results           Results
    └────────┬────────┘
             ▼
    ┌────────────────┐
    │  Answer Builder │
    │ (LLM + sources) │
    └────────────────┘
             ▼
    ┌────────────────┐
    │  Evaluator      │
    │ (accuracy,     │
    │  routing,      │
    │  hallucination) │
    └────────────────┘
```

## Project Structure

```
ExperimentIQ/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application & endpoints
│   ├── router.py            # Question routing logic
│   ├── schemas.py           # Pydantic request/response models
│   ├── structured/          # Structured data access (MLflow API)
│   │   ├── __init__.py
│   │   ├── client.py        # MLflow TrackingServiceClient wrapper
│   │   ├── query.py         # Structured query execution
│   │   └── types.py         # Data types for runs, params, metrics
│   ├── unstructured/        # Unstructured data retrieval
│   │   ├── __init__.py
│   │   ├── client.py        # Qdrant client wrapper
│   │   ├── embed.py         # Embedding generation
│   │   ├── retrieve.py      # Dense + BM25 retrieval
│   │   └── document.py      # Document ingestion & chunking
│   ├── pipeline.py          # End-to-end RAG pipeline
│   └── eval/                # Evaluation suite
│       ├── __init__.py
│       ├── benchmark.py     # Benchmark dataset & runner
│       └── metrics.py       # Accuracy, routing accuracy, hallucination metrics
├── tests/                   # Unit & integration tests
├── scripts/                 # Utility scripts (seed data, run eval, etc.)
├── docker-compose.yml       # Qdrant + API in one command
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+
- MLflow tracking server running with your experiment data
- Qdrant vector database (or let Docker handle it)

### 1. Install dependencies

```bash
cd ExperimentIQ
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -e ".[dev]"
```

### 2. Start infrastructure (optional, via Docker)

```bash
docker-compose up -d qdrant
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your MLflow tracking URI and LLM settings
```

### 4. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Query the API

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "which flow config gave the best validation NLL?"}'
```

### 6. Run evaluations

```bash
python -m app.eval.benchmark run --output results.json
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/ask` | POST | Ask a question, returns answer + sources + route |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API docs (Swagger UI) |

### `/ask` Request Schema

```json
{
  "query": "which flow config gave the best validation NLL?",
  "top_k": 5,
  "route": "auto"  // "auto", "structured", or "unstructured"
}
```

### Response Schema

```json
{
  "answer": "Flow config 'config_v3' achieved the best validation NLL of 0.82",
  "route": "structured",
  "sources": [
    {
      "type": "run",
      "run_id": "abc123",
      "metrics": { "val_nll": 0.82 },
      "params": { "flow_config": "config_v3" }
    }
  ],
  "confidence": 0.92,
  "raw": { ... }
}
```

## Evaluation

ExperimentIQ ships with a benchmark of hand-authored question-answer pairs designed to test:

- **Answer accuracy**: Is the answer correct given the retrieved evidence?
- **Routing accuracy**: Did the router pick the right retrieval path?
- **Hallucination rate**: Did the model answer when it should have said "I don't know"?

Results are reported in the README (or a `results/` directory) with per-category breakdowns.

## Key Design Decisions

1. **Hybrid retrieval over single-modality RAG**: Structured and unstructured data need different retrieval strategies. Routing keeps both fast and accurate.
2. **MIME-based router**: Rule-based routing over query keywords + a lightweight LLM fallback. No heavy LLM call on the hot path.
3. **MLflow API-first**: The system talks to any MLflow Tracking Server via the official Python client. No tight coupling to internal DBs.
4. **Containerized delivery**: `docker-compose up` gets you a running instance with Qdrant + API.

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Structured data | MLflow Tracking API (Python client) |
| Vector DB | Qdrant |
| Embeddings | sentence-transformers (configurable) |
| LLM | Any OpenAI-compatible endpoint |
| Docs | FastAPI Swagger UI |
| Eval | Pytest + custom harness |

## License

MIT

