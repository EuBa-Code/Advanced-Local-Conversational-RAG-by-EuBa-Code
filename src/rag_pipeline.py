import logging
from pathlib import Path

from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank

from config import get_settings, Settings

logger = logging.getLogger(__name__)

RERANKER_MODEL = "ms-marco-TinyBERT-L-2-v2"
RERANKER_CACHE = "models/flashrank"
RETRIEVER_K = 6
RETRIEVER_FETCH_K = 20
RERANKER_TOP_N = 6
MAX_QUERIES = 4
HISTORY_WINDOW = 6


def build_components(settings: Settings):
    """Build embeddings, vector store, reranker, and LLM from settings."""
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=settings.local_embeddings_model,
        cache_folder=settings.hf_models_cache or None,
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

    ranker = Ranker(model_name=RERANKER_MODEL, cache_dir=RERANKER_CACHE)
    compressor = FlashrankRerank(client=ranker, top_n=RERANKER_TOP_N)

    llm = ChatOllama(
        model=settings.local_llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
    )

    return dense_embeddings, sparse_embeddings, vector_store, compressor, llm


def load_system_prompt(filepath: Path) -> str:
    """Read the system prompt file. Raises FileNotFoundError if missing."""
    return filepath.read_text(encoding="utf-8")


def condense_question(llm, chat_history, question: str) -> str:
    """Reformulate a follow-up question into a standalone one using chat history."""
    if not chat_history:
        return question

    prompt = ChatPromptTemplate.from_template(
        "Given the following conversation and a follow-up question, rephrase the follow-up question "
        "to be a standalone question, complete with all necessary context.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Follow Up Question: {question}\n\n"
        "Standalone Question:"
    )
    chain = prompt | llm | StrOutputParser()

    if isinstance(chat_history, list) and chat_history and isinstance(chat_history[0], dict):
        history_str = "\n".join(f"{m['role']}: {m['content']}" for m in chat_history)
    else:
        history_str = "\n".join(f"{type(m).__name__}: {m.content}" for m in chat_history)

    try:
        return chain.invoke({"chat_history": history_str, "question": question}).strip()
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"LLM unavailable while condensing question: {e}")
        return question


def _is_valid_query(text: str) -> bool:
    """Filter out LLM preamble/meta-text that isn't an actual query."""
    lower = text.lower()
    bad_prefixes = ("here are", "sure", "of course", "i'd be", "the following", "below")
    return len(text) > 5 and not any(lower.startswith(p) for p in bad_prefixes)


def generate_multi_queries(llm, query: str) -> list[str]:
    """Generate query variations to improve retrieval recall."""
    prompt = ChatPromptTemplate.from_template(
        "Generate exactly 3 search queries that are variations of the following question. "
        "Each variation must keep all specific names, terms, and numbers from the original. "
        "Only change the phrasing and add synonyms around those key terms. "
        "Output ONLY the 3 queries, one per line. No numbering, no preamble.\n\n"
        "Question: {question}"
    )
    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"question": query})
        lines = [q.strip().lstrip("0123456789.-) ") for q in response.split("\n") if q.strip()]
        queries = [q for q in lines if _is_valid_query(q)]
        if query not in queries:
            queries.insert(0, query)
        return queries[:MAX_QUERIES]
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"LLM unavailable while expanding queries: {e}")
        return [query]


def retrieve_and_rerank(vector_store, compressor, queries: list[str], reference_query: str) -> list[Document]:
    """Run hybrid search for each query, deduplicate, then rerank."""
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K},
    )

    all_docs = []
    for q in queries:
        all_docs.extend(retriever.invoke(q))

    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    if not unique_docs:
        return []

    return compressor.compress_documents(unique_docs, reference_query)


def format_docs(docs: list[Document]) -> str:
    """Format documents into a context string with source attribution."""
    if not docs:
        return "No relevant documents found."

    return "\n\n".join(
        f"[{Path(d.metadata.get('source', 'unknown')).stem}]\n{d.page_content.strip()}"
        for d in docs
    )


def format_history(chat_history, window: int = HISTORY_WINDOW) -> str:
    """Convert recent chat history to a string for the prompt."""
    recent = chat_history[-window:]
    if not recent:
        return ""

    if isinstance(recent[0], dict):
        return "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    return "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in recent
    )
