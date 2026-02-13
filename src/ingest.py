# Script for document loading, splitting, and uploading to Vector Store (Qdrant)

import os
import sys
import logging
from pathlib import Path
from typing import List

from datetime import datetime, timezone
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader 
from langchain_core.documents import Document  
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode 

from config import get_settings

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()

def load_documents(data_path: str) -> List[Document]:
    """
    PHASE 1: DOCUMENT LOADING.
    """
    path = Path(data_path)
    documents: List[Document] = []

    if not path.exists():
        logger.error(f"The path {data_path} does not exist")
        raise FileNotFoundError(f"The path {data_path} does not exist")
    
    if path.is_dir():
        logger.info(f"Loading documents from directory: {data_path}")
        loader = DirectoryLoader(
            path=data_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()
    elif path.is_file():
        logger.info(f"Loading single file: {data_path}")
        loader = TextLoader(str(path), encoding="utf-8")
        documents = loader.load()

    current_date = datetime.now(timezone.utc).isoformat()

    for doc in documents:
        source_path = doc.metadata.get("source", "")
        if os.path.exists(source_path):
            doc.metadata['file_modified_date'] = datetime.fromtimestamp(
                os.path.getmtime(source_path),
                tz=timezone.utc
            ).isoformat()
        
        doc.metadata['ingestion_date'] = current_date
        doc.metadata['filename'] = os.path.basename(source_path)
        doc.metadata['file_type'] = Path(source_path).suffix
        doc.metadata['content_length'] = len(doc.page_content)

    logger.info(f"Loaded {len(documents)} documents with metadata")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    PHASE 2: CHUNKING.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""], 
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


from langchain_huggingface import HuggingFaceEmbeddings 

def create_embedding(chunks):
    """
    PHASE 3 & 4: EMBEDDING AND VECTOR STORE.
    """
    logger.info("Initializing embedding models...")
    
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=settings.local_embeddings_model, 
        cache_folder=settings.hf_models_cache if settings.hf_models_cache else None
    )
    
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    logger.info(f"Uploading to Qdrant collection: {settings.qdrant_collection}")
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        prefer_grpc=False,
        collection_name=settings.qdrant_collection,
        retrieval_mode=RetrievalMode.HYBRID,
        force_recreate=True 
    )
    logger.info("Ingestion completed successfully.")
    return vectorstore

def run_ingestion(data_path: str = None) -> QdrantVectorStore:  
    """Orchestrate the full ingestion pipeline"""
    data_path = data_path or str(settings.data_dir)
    documents = load_documents(data_path)  
    chunks = split_documents(documents)
    vectorstore = create_embedding(chunks)
    return vectorstore 

if __name__ == "__main__":
    logger.info("Starting the ingestion process...")
    run_ingestion()  
