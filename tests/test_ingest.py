import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingest import load_documents, split_documents


class TestLoadDocuments:
    def test_raises_on_missing_path(self):
        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/path")

    def test_loads_single_file(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Test content for ingestion.", encoding="utf-8")
        docs = load_documents(str(f))
        assert len(docs) == 1
        assert "Test content" in docs[0].page_content
        assert docs[0].metadata["filename"] == "doc.txt"
        assert docs[0].metadata["file_type"] == ".txt"
        assert "ingestion_date" in docs[0].metadata

    def test_loads_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("File A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("File B", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 2


class TestSplitDocuments:
    def test_short_doc_stays_single_chunk(self):
        from langchain_core.documents import Document
        docs = [Document(page_content="Short text.", metadata={})]
        chunks = split_documents(docs)
        assert len(chunks) == 1

    def test_long_doc_gets_split(self):
        from langchain_core.documents import Document
        long_text = "word " * 500
        docs = [Document(page_content=long_text, metadata={})]
        chunks = split_documents(docs)
        assert len(chunks) > 1
