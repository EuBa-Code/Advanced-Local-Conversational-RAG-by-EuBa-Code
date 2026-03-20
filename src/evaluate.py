"""
RAGAS evaluation script for the RAG system.

Runs the full pipeline against a test dataset, then scores each response
with Faithfulness and Context Precision using a local Ollama judge model.

Usage: python src/evaluate.py
"""

import asyncio
import gc
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from openai import AsyncOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, Faithfulness

from config import get_settings
from eval_dataset import EVALUATION_DATASET
from rag_pipeline import (
    build_components, generate_multi_queries, retrieve_and_rerank,
    format_docs, load_system_prompt,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results"
OLLAMA_TIMEOUT = 180.0


def run_rag_for_question(llm, vector_store, compressor, qa_chain, question: str) -> dict:
    """Execute the full RAG pipeline for a single evaluation question."""
    queries = generate_multi_queries(llm, question)
    reranked_docs = retrieve_and_rerank(vector_store, compressor, queries, question)
    context = format_docs(reranked_docs)
    response = qa_chain.invoke({"question": question, "context": context})

    return {
        "response": response,
        "retrieved_contexts": [doc.page_content.strip() for doc in reranked_docs],
        "num_docs_retrieved": len(reranked_docs),
    }


async def evaluate_single_sample(faithfulness_scorer, context_precision_scorer, sample: dict) -> dict:
    """Score a single Q&A sample."""
    faithfulness_result = await faithfulness_scorer.ascore(
        user_input=sample["question"],
        response=sample["response"],
        retrieved_contexts=sample["retrieved_contexts"],
    )
    context_precision_result = await context_precision_scorer.ascore(
        user_input=sample["question"],
        reference=sample["ground_truth"],
        retrieved_contexts=sample["retrieved_contexts"],
    )
    return {
        "faithfulness": faithfulness_result.value,
        "context_precision": context_precision_result.value,
    }


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.4f}"


def _clean_score(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return val


async def run_evaluation():
    print("=" * 60)
    print("  RAGAS EVALUATION — RAG System Quality Assessment")
    print("=" * 60)

    settings = get_settings()

    print("\n[1/4] Building RAG pipeline...")
    dense_embeddings, _, vector_store, compressor, llm = build_components(settings)

    system_prompt = load_system_prompt(settings.prompt_path)
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    qa_chain = rag_prompt | llm | StrOutputParser()

    print(f"[2/4] Generating responses for {len(EVALUATION_DATASET)} questions...")
    rag_results = []
    for i, entry in enumerate(EVALUATION_DATASET, 1):
        question = entry["question"]
        print(f"  ({i}/{len(EVALUATION_DATASET)}) {question[:60]}...")
        result = run_rag_for_question(llm, vector_store, compressor, qa_chain, question)
        result["question"] = question
        result["ground_truth"] = entry["ground_truth"]
        result["source_file"] = entry["source_file"]
        rag_results.append(result)

    gc.collect()

    print(f"[3/4] Initializing RAGAS evaluator ({settings.eval_llm_model})...")
    openai_client = AsyncOpenAI(
        base_url=f"{settings.ollama_base_url}/v1",
        api_key="ollama",
        timeout=OLLAMA_TIMEOUT,
    )
    evaluator_llm = llm_factory(model=settings.eval_llm_model, client=openai_client)
    evaluator_embeddings = LangchainEmbeddingsWrapper(dense_embeddings)

    faithfulness_scorer = Faithfulness(llm=evaluator_llm)
    context_precision_scorer = ContextPrecision(llm=evaluator_llm, embeddings=evaluator_embeddings)

    print(f"[4/4] Scoring with RAGAS metrics...")
    evaluation_results = []
    for i, sample in enumerate(rag_results, 1):
        print(f"  Scoring ({i}/{len(rag_results)}) {sample['question'][:50]}...")
        try:
            scores = await evaluate_single_sample(faithfulness_scorer, context_precision_scorer, sample)
            print(f"    - Faithfulness: {_fmt(scores.get('faithfulness'))} | Context Precision: {_fmt(scores.get('context_precision'))}")
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            print(f"    ERROR scoring sample: {e}")
            scores = {"faithfulness": None, "context_precision": None}

        evaluation_results.append({
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "response": sample["response"],
            "source_file": sample["source_file"],
            "num_docs_retrieved": sample["num_docs_retrieved"],
            "scores": scores,
        })

    faithfulness_scores = [s for r in evaluation_results if (s := _clean_score(r["scores"]["faithfulness"])) is not None]
    ctx_precision_scores = [s for r in evaluation_results if (s := _clean_score(r["scores"]["context_precision"])) is not None]

    aggregate = {
        "avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None,
        "avg_context_precision": sum(ctx_precision_scores) / len(ctx_precision_scores) if ctx_precision_scores else None,
        "total_questions": len(evaluation_results),
        "questions_scored": len(faithfulness_scores),
        "model_used": f"{settings.eval_llm_model} (eval) / {settings.local_llm_model} (gen)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"ragas_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {"aggregate_scores": aggregate, "detailed_results": evaluation_results}
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Score':>10}")
    print("-" * 37)
    print(f"{'Faithfulness':<25} {_fmt(aggregate['avg_faithfulness']):>10}")
    print(f"{'Context Precision':<25} {_fmt(aggregate['avg_context_precision']):>10}")
    print(f"{'Questions Evaluated':<25} {aggregate['total_questions']:>10}")

    print(f"\n--- Per-Question Breakdown ---\n")
    for r in evaluation_results:
        print(f"  Q: {r['question'][:55]}...")
        print(f"     Faith: {_fmt(r['scores']['faithfulness'])}  |  CtxPrec: {_fmt(r['scores']['context_precision'])}")
        print(f"     Docs: {r['num_docs_retrieved']}  |  Source: {r['source_file']}")
        print()

    print(f"Full report saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
