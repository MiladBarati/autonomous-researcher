import os
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.rag import RAGPipeline
from config import Config


@pytest.fixture()
def temp_chroma_dir(tmp_path: Any, monkeypatch: Any) -> Any:
    tmp = tmp_path / "chroma"
    os.makedirs(tmp, exist_ok=True)
    monkeypatch.setattr(Config, "CHROMA_PERSIST_DIR", str(tmp), raising=False)
    return tmp


def test_chunk_documents_creates_chunks_with_metadata(
    temp_chroma_dir: Any,  # noqa: ARG001
) -> None:
    # Mock heavy components
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=[[0.1, 0.2]]))
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    docs: list[dict[str, Any]] = [
        {"content": "A " * 600, "title": "T", "url": "u", "source": "web"}
    ]
    chunks: list[dict[str, Any]] = rag.chunk_documents(docs)

    assert len(chunks) > 1
    assert all("id" in c and "text" in c for c in chunks)
    assert all(c["source"] == "web" for c in chunks)


def test_embed_chunks_uses_embedding_model(temp_chroma_dir: Any) -> None:  # noqa: ARG001
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        st_instance = MagicMock()
        st_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        ST.return_value = st_instance
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    chunks: list[dict[str, Any]] = [{"text": "hello"}]
    vectors: list[list[float]] = rag.embed_chunks(chunks)

    assert vectors == [[0.1, 0.2, 0.3]]
    ST.return_value.encode.assert_called_once()


def test_store_documents_adds_to_chroma(temp_chroma_dir: Any) -> None:  # noqa: ARG001
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(
            encode=MagicMock(return_value=np.array([[0.1, 0.2], [0.2, 0.3]]))
        )
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    docs: list[dict[str, Any]] = [
        {"content": "word " * 1200, "title": "Doc1", "url": "u1", "source": "web"},
    ]
    count: int = rag.store_documents(docs)

    assert count > 0
    assert fake_collection.add.called


def test_retrieve_formats_results_and_similarity(
    temp_chroma_dir: Any,  # noqa: ARG001
) -> None:
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=np.array([[0.9, 0.1, 0.0]])))
        fake_collection = MagicMock()
        fake_collection.query.return_value = {
            "ids": [["a", "b"]],
            "documents": [["text a", "text b"]],
            "metadatas": [[{"title": "A"}, {"title": "B"}]],
            "distances": [[0.2, 0.4]],
        }
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    out: list[dict[str, Any]] = rag.retrieve("query", top_k=2)
    assert len(out) == 2
    assert out[0]["similarity_score"] == pytest.approx(0.8)
    assert out[0]["metadata"]["title"] == "A"

    ctx = rag.format_retrieved_context(out)
    assert "[Source 1: A]" in ctx


def test_chunk_documents_handles_empty_content(temp_chroma_dir: Any) -> None:  # noqa: ARG001
    """Test that chunk_documents handles documents with empty content"""
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=[[0.1, 0.2]]))
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    docs: list[dict[str, Any]] = [
        {"content": "", "title": "T", "url": "u", "source": "web"},
        {"title": "T2", "url": "u2", "source": "web"},  # No content field
    ]
    chunks: list[dict[str, Any]] = rag.chunk_documents(docs)

    # Should skip empty content
    assert len(chunks) == 0


def test_chunk_documents_preserves_metadata(temp_chroma_dir: Any) -> None:  # noqa: ARG001
    """Test that chunk_documents preserves all metadata fields"""
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=[[0.1, 0.2]]))
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    docs: list[dict[str, Any]] = [
        {
            "content": "A " * 600,
            "title": "Test Title",
            "url": "http://test",
            "source": "web",
            "authors": ["Author1", "Author2"],
            "published": "2024-01-01",
            "categories": ["cs.AI"],
        }
    ]
    chunks: list[dict[str, Any]] = rag.chunk_documents(docs)

    assert len(chunks) > 0
    assert chunks[0]["title"] == "Test Title"
    assert chunks[0]["url"] == "http://test"
    assert chunks[0]["source"] == "web"
    assert chunks[0]["authors"] == "['Author1', 'Author2']"
    assert chunks[0]["published"] == "2024-01-01"
    assert chunks[0]["categories"] == "['cs.AI']"


def test_store_documents_handles_empty_list(temp_chroma_dir: Any) -> None:  # noqa: ARG001
    """Test that store_documents handles empty document list"""
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=[[0.1, 0.2]]))
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    count: int = rag.store_documents([])
    assert count == 0
    assert not fake_collection.add.called


def test_store_documents_handles_empty_chunks(temp_chroma_dir: Any) -> None:  # noqa: ARG001
    """Test that store_documents handles documents that produce no chunks"""
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=[[0.1, 0.2]]))
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    # Documents with empty content produce no chunks
    docs: list[dict[str, Any]] = [{"content": "", "title": "T", "url": "u", "source": "web"}]
    count: int = rag.store_documents(docs)
    assert count == 0


def test_retrieve_handles_empty_results(temp_chroma_dir: Any) -> None:  # noqa: ARG001
    """Test that retrieve handles empty query results"""
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=np.array([[0.9, 0.1, 0.0]])))
        fake_collection = MagicMock()
        fake_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    out: list[dict[str, Any]] = rag.retrieve("query", top_k=5)
    assert len(out) == 0


def test_retrieve_uses_default_top_k(temp_chroma_dir: Any) -> None:  # noqa: ARG001
    """Test that retrieve uses default top_k when not provided"""
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=np.array([[0.9, 0.1, 0.0]])))
        fake_collection = MagicMock()
        fake_collection.query.return_value = {
            "ids": [["a"]],
            "documents": [["text a"]],
            "metadatas": [[{"title": "A"}]],
            "distances": [[0.2]],
        }
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    rag.retrieve("query")
    # Should use Config.TOP_K_RESULTS (default 5)
    assert fake_collection.query.called
    call_kwargs = fake_collection.query.call_args[1]
    assert call_kwargs["n_results"] == 5


def test_format_retrieved_context_handles_empty_list(temp_chroma_dir: Any) -> None:  # noqa: ARG001
    """Test that format_retrieved_context handles empty chunk list"""
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=[[0.1, 0.2]]))
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    ctx: str = rag.format_retrieved_context([])
    assert ctx == "No relevant context found."


def test_format_retrieved_context_uses_url_when_no_title(
    temp_chroma_dir: Any,  # noqa: ARG001
) -> None:
    """Test that format_retrieved_context uses URL when title is missing"""
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=[[0.1, 0.2]]))
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    chunks: list[dict[str, Any]] = [
        {
            "text": "Test text",
            "metadata": {"url": "http://test.com"},
        }
    ]
    ctx: str = rag.format_retrieved_context(chunks)
    assert "http://test.com" in ctx


def test_get_collection_stats(temp_chroma_dir: Any) -> None:  # noqa: ARG001
    """Test that get_collection_stats returns correct information"""
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=[[0.1, 0.2]]))
        fake_collection = MagicMock()
        fake_collection.count.return_value = 42
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    stats: dict[str, Any] = rag.get_collection_stats()
    assert stats["document_count"] == 42
    assert stats["collection_name"] == "test_collection"
    assert "persist_directory" in stats


def test_clear_collection(temp_chroma_dir: Any) -> None:  # noqa: ARG001
    """Test that clear_collection deletes and recreates collection"""
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=[[0.1, 0.2]]))
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")
        rag.clear_collection()

        assert PC.return_value.delete_collection.called


def test_get_or_create_collection_uses_existing(temp_chroma_dir: Any) -> None:  # noqa: ARG001
    """Test that _get_or_create_collection uses existing collection"""
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=[[0.1, 0.2]]))
        fake_collection = MagicMock()
        PC.return_value.get_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

        assert rag.collection is fake_collection
        assert not PC.return_value.create_collection.called


def test_create_rag_pipeline_factory(temp_chroma_dir: Any) -> None:  # noqa: ARG001
    """Test that create_rag_pipeline factory function works"""
    from agent.rag import create_rag_pipeline

    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=[[0.1, 0.2]]))
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = create_rag_pipeline(collection_name="custom_collection")
        assert rag.collection_name == "custom_collection"


def test_retrieve_handles_invalid_query(temp_chroma_dir: Any) -> None:  # noqa: ARG001
    """Test that retrieve handles invalid query"""
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=[[0.1, 0.2]]))
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    # Empty query should trigger validation error
    out: list[dict[str, Any]] = rag.retrieve("")
    assert len(out) == 0


def test_clear_collection_handles_error(temp_chroma_dir: Any) -> None:  # noqa: ARG001
    """Test that clear_collection handles errors gracefully"""
    with (
        patch("agent.rag.SentenceTransformer") as ST,
        patch("agent.rag.chromadb.PersistentClient") as PC,
    ):
        ST.return_value = MagicMock(encode=MagicMock(return_value=[[0.1, 0.2]]))
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection
        PC.return_value.delete_collection.side_effect = Exception("delete error")

        rag = RAGPipeline(collection_name="test_collection")
        # Should not raise, but log error
        rag.clear_collection()
        # Collection should still exist (recreated)
        assert rag.collection is not None
