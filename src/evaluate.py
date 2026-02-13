"""
RAGAS Evaluation Script for the RAG System using Local Ollama.

Runs the full RAG pipeline (retrieval + reranking + generation) against
a test dataset, then evaluates each response with two RAGAS metrics:
- Faithfulness: Is the response grounded in the retrieved documents?
- Context Precision: Were the retrieved documents actually useful?

Usage:
    python src/evaluate.py
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# LangChain Imports for Generation
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Reranking
from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever

# RAGAS Imports
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, ContextPrecision

from config import get_settings
from eval_dataset import EVALUATION_DATASET

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results"
RERANKER_MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"
RERANKER_CACHE_DIR = "models/flashrank"
RETRIEVER_TOP_K = 6
RETRIEVER_FETCH_K = 20
RERANKER_TOP_N = 4


def build_rag_pipeline(settings):
    """Build and return the retrieval chain components (retriever, LLM, prompt)."""
    
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=settings.local_embeddings_model,
        cache_folder=settings.hf_models_cache if settings.hf_models_cache else None,
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        prefer_grpc=False,
        collection_name=settings.qdrant_collection,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_TOP_K, "fetch_k": RETRIEVER_FETCH_K},
    )

    ranker_client = Ranker(
        model_name=RERANKER_MODEL_NAME, cache_dir=RERANKER_CACHE_DIR
    )
    compressor = FlashrankRerank(client=ranker_client, top_n=RERANKER_TOP_N)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    print(f"    Using Local LLM for Generation: {settings.local_llm_model}")
    llm = ChatOllama(
        model=settings.local_llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
    )

    system_prompt = settings.prompt_path.read_text(encoding="utf-8")
    rag_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "Context:\n{context}\n\nQuestion: {question}")]
    )

    qa_chain = rag_prompt | llm | StrOutputParser()

    return llm, dense_embeddings, sparse_embeddings, vector_store, ranker_client, qa_chain

def run_rag_for_question(llm, dense_embeddings, sparse_embeddings, vector_store, ranker, qa_chain, question: str) -> dict:
    """Execute the full RAG pipeline (Multi-Query + Reranking) for evaluation."""
    
    # logic from app.py
    # 1. Expand query (Multi-Query)
    from langchain_core.prompts import ChatPromptTemplate as CPT
    expansion_prompt = CPT.from_template(
        "Generate 3 different versions of the following question to improve retrieval: {question}. "
        "Return only the questions separated by a comma."
    )
    expansion_chain = expansion_prompt | llm | StrOutputParser()
    try:
        resp = expansion_chain.invoke({"question": question})
        queries = [q.strip() for q in resp.replace('\n', ',').split(',') if q.strip()]
        if question not in queries: queries.insert(0, question)
        queries = queries[:4]
    except:
        queries = [question]

    # 2. Retrieval
    base_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})
    all_docs = []
    for q in queries:
        all_docs.extend(base_retriever.invoke(q))
    
    # 3. Deduplicate
    seen = set()
    unique_docs = []
    for d in all_docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique_docs.append(d)

    # 4. Rerank
    from langchain_community.document_compressors import FlashrankRerank
    compressor = FlashrankRerank(client=ranker, top_n=4)
    reranked_docs = compressor.compress_documents(unique_docs, question)
    
    formatted_context = "\n\n".join([f"(Source: {d.metadata.get('source', 'unknown')})\n{d.page_content.strip()}" for d in reranked_docs])

    response = qa_chain.invoke({"question": question, "context": formatted_context})

    return {
        "response": response,
        "retrieved_contexts": [doc.page_content.strip() for doc in reranked_docs],
        "num_docs_retrieved": len(reranked_docs),
    }


async def evaluate_single_sample(
    faithfulness_scorer, context_precision_scorer, sample: dict
) -> dict:
    """Score a single Q&A sample with all RAGAS metrics."""
    
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


async def run_evaluation():
    """Orchestrate the full evaluation pipeline."""
    print("=" * 60)
    print("  RAGAS EVALUATION (Local Ollama) — RAG System Quality Assessment")
    print("=" * 60)

    settings = get_settings()

    # --- Build RAG Pipeline ---
    print("\n[1/4] Building RAG pipeline...")
    llm, dense_embeddings, sparse_embeddings, vector_store, ranker, qa_chain = build_rag_pipeline(settings)

    # --- Generate Responses ---
    print(f"[2/4] Generating responses for {len(EVALUATION_DATASET)} questions...")
    rag_results = []
    for i, entry in enumerate(EVALUATION_DATASET, start=1):
        question = entry["question"]
        print(f"  ({i}/{len(EVALUATION_DATASET)}) {question[:60]}...")

        result = run_rag_for_question(llm, dense_embeddings, sparse_embeddings, vector_store, ranker, qa_chain, question)
        result["question"] = question
        result["ground_truth"] = entry["ground_truth"]
        result["source_file"] = entry["source_file"]
        rag_results.append(result)

    # --- Initialize RAGAS Metrics ---
    print(f"[3/4] Initializing RAGAS evaluator with Local LLM ({settings.local_llm_model})...")
    
    # Use LangChain wrapper for RAGAS (ignoring deprecation for stability)
    from ragas.llms import LangchainLLMWrapper
    
    evaluator_llm_lc = ChatOllama(
        model=settings.local_llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
    )
    
    evaluator_llm = LangchainLLMWrapper(evaluator_llm_lc)

    faithfulness_scorer = Faithfulness(llm=evaluator_llm)
    context_precision_scorer = ContextPrecision(llm=evaluator_llm)

    # --- Evaluate ---
    print(f"[4/4] Evaluating with RAGAS metrics...")
    evaluation_results = []

    for i, sample in enumerate(rag_results, start=1):
        print(f"  Scoring ({i}/{len(rag_results)}) {sample['question'][:50]}...")

        try:
            scores = await evaluate_single_sample(
                faithfulness_scorer, context_precision_scorer, sample
            )
        except Exception as e:
            print(f"    ERROR scoring sample: {e}")
            scores = {"faithfulness": None, "context_precision": None}

        evaluation_results.append(
            {
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "response": sample["response"],
                "source_file": sample["source_file"],
                "num_docs_retrieved": sample["num_docs_retrieved"],
                "scores": scores,
            }
        )

    # --- Aggregate and Save ---
    faithfulness_scores = [
        r["scores"]["faithfulness"]
        for r in evaluation_results
        if r["scores"]["faithfulness"] is not None
    ]
    context_precision_scores = [
        r["scores"]["context_precision"]
        for r in evaluation_results
        if r["scores"]["context_precision"] is not None
    ]

    aggregate = {
        "avg_faithfulness": (
            sum(faithfulness_scores) / len(faithfulness_scores)
            if faithfulness_scores
            else None
        ),
        "avg_context_precision": (
            sum(context_precision_scores) / len(context_precision_scores)
            if context_precision_scores
            else None
        ),
        "total_questions": len(evaluation_results),
        "model_used": f"Local Ollama ({settings.local_llm_model})",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save full results to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"ragas_eval_{timestamp_label}.json"

    report = {"aggregate_scores": aggregate, "detailed_results": evaluation_results}

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Print Results Table ---
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
        print(f"     Faith: {_fmt(r['scores']['faithfulness'])}  |  "
              f"CtxPrec: {_fmt(r['scores']['context_precision'])}")
        print(f"     Docs: {r['num_docs_retrieved']}  |  Source: {r['source_file']}")
        print()

    print(f"Full report saved to: {output_path}")
    print("=" * 60)


def _fmt(value) -> str:
    """Format a score value for display."""
    if value is None:
        return "N/A"
    return f"{value:.4f}"


if __name__ == "__main__":
    asyncio.run(run_evaluation())
