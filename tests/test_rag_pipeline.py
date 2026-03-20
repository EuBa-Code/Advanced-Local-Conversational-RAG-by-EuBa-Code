import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from rag_pipeline import format_docs, format_history, load_system_prompt


class TestFormatDocs:
    def test_empty_list_returns_fallback(self):
        assert format_docs([]) == "No relevant documents found."

    def test_single_doc(self):
        doc = Document(page_content="Hello world", metadata={"source": "test.txt"})
        result = format_docs([doc])
        assert "Hello world" in result
        assert "test.txt" in result

    def test_multiple_docs_separated(self):
        docs = [
            Document(page_content="First", metadata={"source": "a.txt"}),
            Document(page_content="Second", metadata={"source": "b.txt"}),
        ]
        result = format_docs(docs)
        assert "First" in result
        assert "Second" in result
        assert "\n\n" in result

    def test_missing_source_uses_unknown(self):
        doc = Document(page_content="content", metadata={})
        result = format_docs([doc])
        assert "unknown" in result


class TestFormatHistory:
    def test_empty_history(self):
        assert format_history([]) == ""

    def test_dict_format(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = format_history(history)
        assert "user: hi" in result
        assert "assistant: hello" in result

    def test_message_format(self):
        history = [
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
        result = format_history(history)
        assert "User: hi" in result
        assert "AI: hello" in result

    def test_window_limits_messages(self):
        history = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        result = format_history(history, window=3)
        assert "msg7" in result
        assert "msg8" in result
        assert "msg9" in result
        assert "msg0" not in result


class TestLoadSystemPrompt:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "prompt.txt"
        f.write_text("You are a helpful assistant.", encoding="utf-8")
        assert load_system_prompt(f) == "You are a helpful assistant."

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_system_prompt(tmp_path / "nonexistent.txt")
