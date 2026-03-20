import sys
import streamlit as st
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import get_settings
from rag_pipeline import (
    build_components, load_system_prompt, condense_question,
    generate_multi_queries, retrieve_and_rerank, format_docs, format_history,
)

st.set_page_config(page_title="AGS Contextual Chat", page_icon="🤖", layout="wide")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def init_components():
    settings = get_settings()
    dense, sparse, vector_store, compressor, llm = build_components(settings)
    system_prompt = load_system_prompt(settings.prompt_path)
    return settings, vector_store, compressor, llm, system_prompt


if "messages" not in st.session_state:
    st.session_state.messages = []

settings, vector_store, compressor, llm, system_prompt = init_components()

st.title("🤖 Aetheria Global Solutions - AI Support")
st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about AGS..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        with st.status("Searching knowledge base...", expanded=False) as status:
            standalone = condense_question(llm, st.session_state.messages[:-1], prompt)
            st.write(f"🔍 Normalized query: {standalone}")

            queries = generate_multi_queries(llm, standalone)
            st.write(f"🔎 Expanding to: {', '.join(queries)}")

            st.write("🎯 Reranking results...")
            reranked = retrieve_and_rerank(vector_store, compressor, queries, standalone)
            status.update(label="Information retrieved!", state="complete")

        context = format_docs(reranked)
        history_str = format_history(st.session_state.messages)

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nHistory:\n{history}\n\nQuestion: {question}"),
        ])
        chain = rag_prompt | llm | StrOutputParser()

        for chunk in chain.stream({"context": context, "history": history_str, "question": prompt}):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

        with st.expander("Show Sources"):
            for d in reranked:
                st.info(f"**Source**: {d.metadata.get('source', 'unknown')}\n\n{d.page_content[:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": full_response})

if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🛑 Exit & Shutdown Server", help="Stop the RAG system and free RAM/CPU."):
    st.sidebar.error("System shutting down... You can close this tab.")
    sys.exit(0)
