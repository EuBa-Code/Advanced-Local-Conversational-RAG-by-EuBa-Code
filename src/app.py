import sys
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from config import get_settings
from rag_pipeline import (
    build_components, load_system_prompt, condense_question,
    generate_multi_queries, retrieve_and_rerank, format_docs, format_history,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting Application (Conversational RAG + Multi-Query)...")
    settings = get_settings()
    logger.info(f"Local LLM Model: {settings.local_llm_model}")

    system_prompt_content = load_system_prompt(settings.prompt_path)
    _, _, vector_store, compressor, llm = build_components(settings)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_content),
        ("human", "Retrieved Context:\n{context}\n\nConversation History:\n{chat_history_str}\n\nUser Question: {question}"),
    ])
    qa_chain = rag_prompt | llm | StrOutputParser()

    chat_history: list[BaseMessage] = []
    logger.info("Setup completed. (Type 'reset' to clear memory)")

    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ("esci", "exit", "quit"):
                logger.info("Goodbye!")
                break
            if query.lower() == "reset":
                chat_history = []
                logger.info("Memory cleared.")
                continue

            standalone_query = condense_question(llm, chat_history, query)
            if standalone_query != query:
                logger.info(f"Question reformulated to: {standalone_query}")

            queries = generate_multi_queries(llm, standalone_query)
            reranked_docs = retrieve_and_rerank(vector_store, compressor, queries, standalone_query)
            formatted_context = format_docs(reranked_docs)
            history_str = format_history(chat_history)

            print("\nAndrea (AI): ", end="", flush=True)
            full_response = ""
            for chunk in qa_chain.stream({
                "question": query,
                "context": formatted_context,
                "chat_history_str": history_str,
            }):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")

            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=full_response))

            if logger.isEnabledFor(logging.DEBUG) and reranked_docs:
                logger.debug("-" * 40)
                for idx, doc in enumerate(reranked_docs, 1):
                    source = doc.metadata.get("source", "unknown")
                    snippet = doc.page_content[:80].replace("\n", " ")
                    logger.debug(f"{idx}. {source} - \"{snippet}...\"")
                logger.debug("-" * 40)

        except KeyboardInterrupt:
            logger.info("Exit requested via keyboard.")
            break


if __name__ == "__main__":
    main()
