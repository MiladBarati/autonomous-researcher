import os
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from agent.rag import RAGPipeline
from config import Config


@pytest.fixture()
def temp_chroma_dir(tmp_path, monkeypatch):
    tmp = tmp_path / "chroma"
    os.makedirs(tmp, exist_ok=True)
    monkeypatch.setattr(Config, "CHROMA_PERSIST_DIR", str(tmp), raising=False)
    return tmp


def test_chunk_documents_creates_chunks_with_metadata(temp_chroma_dir, monkeypatch):
    # Mock heavy components
    with patch("agent.rag.SentenceTransformer") as ST, patch("agent.rag.chromadb.PersistentClient") as PC:
        ST.return_value = MagicMock(encode=MagicMock(return_value=[[0.1, 0.2]]))
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    docs = [{"content": "A " * 600, "title": "T", "url": "u", "source": "web"}]
    chunks = rag.chunk_documents(docs)

    assert len(chunks) > 1
    assert all("id" in c and "text" in c for c in chunks)
    assert all(c["source"] == "web" for c in chunks)


def test_embed_chunks_uses_embedding_model(temp_chroma_dir, monkeypatch):
    with patch("agent.rag.SentenceTransformer") as ST, patch("agent.rag.chromadb.PersistentClient") as PC:
        st_instance = MagicMock()
        st_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        ST.return_value = st_instance
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    chunks = [{"text": "hello"}]
    vectors = rag.embed_chunks(chunks)

    assert vectors == [[0.1, 0.2, 0.3]]
    ST.return_value.encode.assert_called_once()


def test_store_documents_adds_to_chroma(temp_chroma_dir, monkeypatch):
    with patch("agent.rag.SentenceTransformer") as ST, patch("agent.rag.chromadb.PersistentClient") as PC:
        ST.return_value = MagicMock(encode=MagicMock(return_value=np.array([[0.1, 0.2], [0.2, 0.3]])))
        fake_collection = MagicMock()
        PC.return_value.get_collection.side_effect = Exception("no collection")
        PC.return_value.create_collection.return_value = fake_collection

        rag = RAGPipeline(collection_name="test_collection")

    docs = [
        {"content": "word " * 1200, "title": "Doc1", "url": "u1", "source": "web"},
    ]
    count = rag.store_documents(docs)

    assert count > 0
    assert fake_collection.add.called


def test_retrieve_formats_results_and_similarity(temp_chroma_dir, monkeypatch):
    with patch("agent.rag.SentenceTransformer") as ST, patch("agent.rag.chromadb.PersistentClient") as PC:
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

    out = rag.retrieve("query", top_k=2)
    assert len(out) == 2
    assert out[0]["similarity_score"] == pytest.approx(0.8)
    assert out[0]["metadata"]["title"] == "A"

    ctx = rag.format_retrieved_context(out)
    assert "[Source 1: A]" in ctx


