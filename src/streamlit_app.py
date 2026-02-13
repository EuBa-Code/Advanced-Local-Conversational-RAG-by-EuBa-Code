import streamlit as st
import logging
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank
from config import get_settings

# Page config
st.set_page_config(page_title="AGS Contextual Chat", page_icon="🤖", layout="wide")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def init_components():
    """Initialize fixed components once and cache them."""
    settings = get_settings()
    
    # 1. Embeddings
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=settings.local_embeddings_model,
        cache_folder=settings.hf_models_cache if settings.hf_models_cache else None
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    # 2. Vector Store
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
        retrieval_mode=RetrievalMode.HYBRID
    )
    
    # 3. Reranker
    ranker_client = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="models/flashrank")
    compressor = FlashrankRerank(client=ranker_client, top_n=4)
    
    # 4. LLM
    llm = ChatOllama(model=settings.local_llm_model, base_url=settings.ollama_base_url, temperature=0.0)
    
    # 5. Prompt content (system prompt)
    system_prompt = settings.prompt_path.read_text(encoding="utf-8")
    
    return settings, vector_store, compressor, llm, system_prompt

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load components
settings, vector_store, compressor, llm, system_prompt = init_components()

def condense_question(llm, chat_history, question):
    if not chat_history:
        return question
    condense_prompt = ChatPromptTemplate.from_template(
        "Given the conversation history and a follow-up question, rephrase it to be standalone: "
        "\nHistory: {history}\nQuestion: {question}\nStandalone Question:"
    )
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])
    chain = condense_prompt | llm | StrOutputParser()
    return chain.invoke({"history": history_str, "question": question}).strip()

def generate_multi_queries(llm, query):
    prompt = ChatPromptTemplate.from_template(
        "Generate 3 diverse versions of this search query to improve retrieval: {question}. "
        "Return results separated by commas."
    )
    chain = prompt | llm | StrOutputParser()
    resp = chain.invoke({"question": query})
    queries = [q.strip() for q in resp.split(",") if q.strip()]
    if query not in queries: queries.insert(0, query)
    return queries[:4]

# UI Layout
st.title("🤖 Aetheria Global Solutions - AI Support")
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about AGS..."):
    # Clear previous response
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # 1. Processing
        with st.status("Searching knowledge base...", expanded=False) as status:
            standalone = condense_question(llm, st.session_state.messages[:-1], prompt)
            st.write(f"🔍 Normalized query: {standalone}")
            
            queries = generate_multi_queries(llm, standalone)
            st.write(f"🔎 Expanding to: {', '.join(queries)}")
            
            # Retrieval
            retriever = vector_store.as_retriever(search_kwargs={'k': 6})
            docs = []
            for q in queries:
                docs.extend(retriever.invoke(q))
            
            # Deduplicate
            seen = set()
            unique_docs = []
            for d in docs:
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    unique_docs.append(d)
            
            # Rerank
            st.write("🎯 Reranking results...")
            reranked = compressor.compress_documents(unique_docs, standalone)
            status.update(label="Information retrieved!", state="complete")

        # 2. Generation
        context = "\n\n".join([f"(Source: {d.metadata.get('source', 'unknown')})\n{d.page_content}" for d in reranked])
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])
        
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nHistory:\n{history}\n\nQuestion: {question}")
        ])
        
        chain = rag_prompt | llm | StrOutputParser()
        
        for chunk in chain.stream({"context": context, "history": history_str, "question": prompt}):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        
        # Sources in sidebar or expander
        with st.expander("Show Sources"):
            for d in reranked:
                st.info(f"**Source**: {d.metadata.get('source', 'unknown')}\n\n{d.page_content[:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": full_response})

if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

# ---------------------------------------------------------------------------
# Shutdown System
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
if st.sidebar.button("🛑 Exit & Shutdown Server", help="Click here to completely stop the RAG system and free up RAM/CPU."):
    st.sidebar.error("System shutting down... You can close this tab.")
    import os
    import time
    time.sleep(1) # Give time to show the message
    os._exit(0)
