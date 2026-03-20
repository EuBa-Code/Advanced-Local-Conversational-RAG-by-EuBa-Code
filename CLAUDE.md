# CLAUDE.md - Project Guide

## Setup
- Python >=3.12, managed with `uv`
- Run `uv sync` to install dependencies
- Run `uv run pytest tests/ -v` to run tests
- Qdrant (cloud or local) + Ollama for local LLM

## Architecture
- `src/rag_pipeline.py` — shared RAG logic (retrieval, reranking, query expansion, formatting)
- `src/app.py` — CLI chat interface
- `src/streamlit_app.py` — Web UI (Streamlit)
- `src/evaluate.py` — RAGAS evaluation pipeline
- `src/ingest.py` — document ingestion into Qdrant
- `src/config.py` — centralized settings from .env
- `src/eval_dataset.py` — test Q&A pairs for evaluation
- `tests/` — pytest unit tests

## Conventions
- All shared RAG logic lives in `rag_pipeline.py` — don't duplicate in app/streamlit/evaluate
- Use specific exceptions, not bare `except` or `except Exception`
- Constants at module level with UPPER_SNAKE_CASE
- No AI-style comments (no numbered steps, no docstrings that repeat function names)
