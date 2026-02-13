import sys
import logging
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Libraries for Reranking and context compression
from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank

from config import get_settings

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Settings initialization
settings = get_settings()

def format_docs(docs: list[Document]) -> str:
    """Formats retrieved documents into a readable string for the LLM."""
    if not docs:
        logger.warning("No documents found for formatting.")
        return "No relevant documents found."

    unique_docs = []
    seen = set()
    for doc in docs:
        content_hash = hash(doc.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)

    return "\n\n".join(
        f"(Source: {d.metadata.get('source', 'unknown')})\n{d.page_content.strip()}"
        for d in unique_docs
    )

def load_system_prompt(filepath) -> str:
    """Loads system instructions (System Prompt) from an external file."""
    try:
        return filepath.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error(f"Error loading prompt: '{filepath}'")
        sys.exit(1)

def condense_question(llm, chat_history: list[BaseMessage], question: str) -> str:
    """
    If there is a history, reformulates the current question to be standalone,
    incorporating context from previous turns.
    """
    if not chat_history:
        return question

    condense_prompt = ChatPromptTemplate.from_template(
        "Given the following conversation and a follow-up question, rephrase the follow-up question "
        "to be a standalone question, complete with all necessary context.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Follow Up Question: {question}\n\n"
        "Standalone Question:"
    )
    
    chain = condense_prompt | llm | StrOutputParser()
    
    # Convert message list to string for the prompt
    history_str = "\n".join([f"{type(m).__name__}: {m.content}" for m in chat_history])
    
    try:
        standalone_question = chain.invoke({"chat_history": history_str, "question": question})
        return standalone_question.strip()
    except Exception as e:
        logger.error(f"Error condensing question: {e}")
        return question

def generate_multi_queries(llm, original_query: str) -> list[str]:
    """Generates 3 variations of the original question using the LLM."""
    prompt = ChatPromptTemplate.from_template(
        "You are an AI assistant expert in information retrieval.\n"
        "Generate 3 different versions of the following user question to improve retrieval in a vector database.\n"
        "The variations should cover different aspects or terminologies but maintain the same intent.\n"
        "Return ONLY the 3 questions separated by a comma.\n"
        "Original Question: {question}"
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"question": original_query})
        queries = [q.strip() for q in response.replace('\n', ',').split(',') if q.strip()]
        if original_query not in queries:
            queries.insert(0, original_query)
        return queries[:4]
    except Exception as e:
        logger.error(f"Error generating query: {e}")
        return [original_query]

def main() -> None:
    logger.info("Starting Application (Conversational RAG + Multi-Query)...")
    logger.info(f"Local LLM Model: {settings.local_llm_model}")

    system_prompt_content = load_system_prompt(settings.prompt_path) 

    # 1. EMBEDDINGS & VECTOR STORE
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=settings.local_embeddings_model,
        cache_folder=settings.hf_models_cache if settings.hf_models_cache else None
    )  
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=dense_embeddings,  
        sparse_embedding=sparse_embeddings,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        prefer_grpc=False,
        collection_name=settings.qdrant_collection,
        retrieval_mode=RetrievalMode.HYBRID
    )

    base_retriever = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 6, 'fetch_k': 20}
    )

    # 2. RERANKER
    ranker_client = Ranker(
        model_name="ms-marco-TinyBERT-L-2-v2", 
        cache_dir="models/flashrank"
    )
    compressor = FlashrankRerank(client=ranker_client, top_n=4)

    # 3. LOCAL LLM
    llm_local = ChatOllama(
        model=settings.local_llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.0
    )

    # 4. Final Prompt with History
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_content),
        ("human", "Retrieved Context:\n{context}\n\nConversation History:\n{chat_history_str}\n\nUser Question: {question}")
    ])
    
    qa_chain = rag_prompt | llm_local | StrOutputParser()

    # --- CHAT STATE ---
    chat_history: list[BaseMessage] = [] 

    logger.info("Setup completed. (Type 'reset' to clear memory)")
    
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ["esci", "exit", "quit"]:
                logger.info("Goodbye!")
                break
            if query.lower() == "reset":
                chat_history = []
                logger.info("Memory cleared.")
                continue
            
            # 1. Question Reformulation
            standalone_query = condense_question(llm_local, chat_history, query)
            if standalone_query != query:
                logger.info(f"Question reformulated to: {standalone_query}")
            
            # 2. Multi-Query Expansion
            queries = generate_multi_queries(llm_local, standalone_query)
            
            # 3. Retrieval + Deduplication
            all_docs = []
            for q in queries:
                all_docs.extend(base_retriever.invoke(q))
            
            seen_content = set()
            unique_docs = []
            for doc in all_docs:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_docs.append(doc)

            # 4. Reranking
            reranked_docs = compressor.compress_documents(unique_docs, standalone_query)
            formatted_context = format_docs(reranked_docs)
            
            # 5. Response Generation
            history_str = "\n".join([f"{('User' if isinstance(m, HumanMessage) else 'AI')}: {m.content}" for m in chat_history[-6:]])
            
            print("\nAndrea (AI): ", end="", flush=True)
            full_response = ""
            for chunk in qa_chain.stream({
                "question": query,
                "context": formatted_context,
                "chat_history_str": history_str
            }):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")

            # 6. Memory Update
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=full_response))

            # --- SOURCE DEBUG ---
            logger.debug("-" * 40)
            if reranked_docs:
                for idx, doc in enumerate(reranked_docs, 1):
                    source = doc.metadata.get('source', 'unknown')
                    snippet = doc.page_content[:80].replace('\n', ' ')
                    logger.debug(f"{idx}. {source} - \"{snippet}...\"")
            logger.debug("-" * 40)

        except KeyboardInterrupt:
            logger.info("Exit requested via keyboard.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
