"""
RAG Pipeline for Document Storage and Retrieval

Implements chunking, embedding, vector storage, and semantic retrieval
using ChromaDB and sentence-transformers.
"""

import hashlib
from datetime import datetime
from typing import Any, TypedDict, cast

import chromadb
from chromadb import Collection
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from agent.logger import get_logger
from agent.validation import ValidationError, validate_query
from config import Config


# TypedDict for ChromaDB query results
# ChromaDB returns a dict with these keys when querying
class ChromaQueryResult(TypedDict, total=False):
    """Type definition for ChromaDB query results"""

    ids: list[list[str]]
    documents: list[list[str]]
    metadatas: list[list[dict[str, Any]]]
    distances: list[list[float]]


try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback for older langchain versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]

logger = get_logger("rag")


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for document management.

    Handles:
    - Document chunking
    - Embedding generation
    - Vector storage in ChromaDB
    - Semantic retrieval
    """

    def __init__(self, collection_name: str | None = None) -> None:
        """
        Initialize RAG pipeline.

        Args:
            collection_name: Name for ChromaDB collection (default from config)
        """
        self.collection_name: str = collection_name or Config.COLLECTION_NAME

        # Initialize text splitter
        self.text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Initialize embedding model
        logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
        self.embedding_model: SentenceTransformer = SentenceTransformer(Config.EMBEDDING_MODEL)

        # Initialize ChromaDB client
        # Note: Using type: ignore because chromadb doesn't have complete type stubs
        self.client: Any = chromadb.PersistentClient(
            path=Config.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Create or get collection
        self.collection: Collection = self._get_or_create_collection()

    def _is_corruption_error(self, error: Exception) -> bool:
        """Check if an error indicates ChromaDB collection corruption."""
        error_msg = str(error).lower()
        return "_type" in error_msg

    def _create_collection(self) -> Collection:
        """Create a new ChromaDB collection."""
        collection = self.client.create_collection(  # type: ignore[attr-defined]
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Created collection: {self.collection_name}")
        return cast(Collection, collection)

    def _delete_collection_safe(self) -> None:
        """Safely delete collection, ignoring errors if it doesn't exist."""
        try:
            self.client.delete_collection(name=self.collection_name)  # type: ignore[attr-defined]
            logger.info(f"Deleted collection: {self.collection_name}")
        except chromadb.errors.InvalidCollectionException:
            logger.debug(f"Collection didn't exist: {self.collection_name}")
        except Exception as e:
            logger.debug(f"Could not delete collection: {e}")

    def _recover_collection(self, error: Exception) -> Collection:
        """
        Attempt to recover from a corrupted or problematic collection.

        Args:
            error: The exception that triggered recovery

        Returns:
            Newly created collection

        Raises:
            RuntimeError: If recovery fails
        """
        is_corruption = self._is_corruption_error(error)
        error_type = type(error).__name__

        log_msg = (
            "Detected corrupted ChromaDB collection configuration"
            if is_corruption
            else "ChromaDB error accessing collection"
        )
        logger.warning(
            f"{log_msg} ({error_type}): {error}. "
            f"Attempting to delete and recreate collection: {self.collection_name}"
        )

        self._delete_collection_safe()

        try:
            return self._create_collection()
        except Exception as create_error:
            logger.error(f"Failed to recreate collection: {create_error}", exc_info=True)
            raise RuntimeError(
                f"ChromaDB collection is corrupted and could not be recovered. "
                f"Please delete the database directory '{Config.CHROMA_PERSIST_DIR}' and restart."
            ) from create_error

    def _get_or_create_collection(self) -> Collection:
        """
        Get existing collection or create new one.

        Handles corruption recovery automatically.

        Returns:
            ChromaDB collection instance
        """
        try:
            collection = self.client.get_collection(name=self.collection_name)  # type: ignore[attr-defined]
            logger.debug(f"Using existing collection: {self.collection_name}")
            return cast(Collection, collection)
        except chromadb.errors.InvalidCollectionException:
            return self._create_collection()
        except Exception as e:
            if isinstance(e, chromadb.errors.ChromaError | KeyError) or self._is_corruption_error(
                e
            ):
                return self._recover_collection(e)
            logger.error(f"Unexpected error accessing collection: {e}", exc_info=True)
            raise

    def _create_chunk_id(self, url: str, index: int, text_preview: str) -> str:
        """Generate a unique chunk ID."""
        return hashlib.md5(
            f"{url}{index}{text_preview}".encode(), usedforsecurity=False
        ).hexdigest()

    def _create_chunk_metadata(
        self, doc: dict[str, Any], chunk_index: int, total_chunks: int
    ) -> dict[str, Any]:
        """Create metadata dictionary for a chunk."""
        metadata = {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "source": doc.get("source", "unknown"),
            "url": doc.get("url", ""),
            "title": doc.get("title", ""),
            "timestamp": datetime.now().isoformat(),
        }

        # Add optional metadata fields
        for key in ["authors", "published", "categories"]:
            if key in doc:
                metadata[key] = str(doc[key])

        return metadata

    def _chunk_document(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        """Split a single document into chunks."""
        content = doc.get("content", "")
        if not content:
            return []

        text_chunks = self.text_splitter.split_text(content)
        chunks = []

        for i, chunk_text in enumerate(text_chunks):
            chunk_id = self._create_chunk_id(doc.get("url", ""), i, chunk_text[:100])
            metadata = self._create_chunk_metadata(doc, i, len(text_chunks))

            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    **metadata,
                }
            )

        return chunks

    def chunk_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents with 'content' field

        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        for doc in documents:
            chunks.extend(self._chunk_document(doc))
        return chunks

    def _convert_embeddings_to_list(self, embeddings: Any) -> list[list[float]]:
        """Convert embeddings to list of lists format."""
        if hasattr(embeddings, "tolist"):
            return cast(list[list[float]], embeddings.tolist())
        return cast(list[list[float]], embeddings)

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[list[float]]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of embedding vectors
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        return self._convert_embeddings_to_list(embeddings)

    def _prepare_chromadb_metadata(self, chunk: dict[str, Any]) -> dict[str, Any]:
        """Prepare chunk metadata for ChromaDB (excludes id and text, ensures proper types)."""
        metadata = {k: v for k, v in chunk.items() if k not in ["id", "text"]}
        # ChromaDB requires all metadata values to be strings, ints, floats, or bools
        return {
            k: str(v) if not isinstance(v, str | int | float | bool) else v
            for k, v in metadata.items()
        }

    def _prepare_chromadb_data(
        self, chunks: list[dict[str, Any]]
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """Prepare data for ChromaDB storage."""
        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [self._prepare_chromadb_metadata(chunk) for chunk in chunks]
        return ids, texts, metadatas

    def store_documents(self, documents: list[dict[str, Any]]) -> int:
        """
        Process, embed, and store documents in vector database.

        Args:
            documents: List of documents to store

        Returns:
            Number of chunks stored
        """
        if not documents:
            return 0

        logger.debug(f"Chunking {len(documents)} documents...")
        chunks = self.chunk_documents(documents)
        if not chunks:
            return 0

        logger.debug(f"Created {len(chunks)} chunks, generating embeddings...")
        embeddings = self.embed_chunks(chunks)

        ids, texts, metadatas = self._prepare_chromadb_data(chunks)

        logger.debug("Storing in ChromaDB...")
        self.collection.add(
            ids=ids,
            embeddings=cast(Any, embeddings),
            documents=texts,
            metadatas=cast(Any, metadatas),
        )

        logger.info(f"Successfully stored {len(chunks)} chunks")
        return len(chunks)

    def _generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a query string."""
        embeddings = self.embedding_model.encode(
            [query], show_progress_bar=False, convert_to_numpy=True
        )
        embedding_list = self._convert_embeddings_to_list(embeddings)
        return embedding_list[0]

    def _format_retrieval_results(self, results: ChromaQueryResult) -> list[dict[str, Any]]:
        """Format ChromaDB query results into chunk dictionaries."""
        if not results.get("ids") or not results["ids"][0]:
            return []

        retrieved_chunks = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, _ in enumerate(ids):
            retrieved_chunks.append(
                {
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "similarity_score": 1 - distances[i],  # Convert distance to similarity
                    "rank": i + 1,
                }
            )

        return retrieved_chunks

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with metadata
        """
        try:
            query = validate_query(query)
        except ValidationError as e:
            logger.error(f"Invalid retrieval query: {e}")
            return []

        top_k_value = top_k or Config.TOP_K_RESULTS
        query_embedding = self._generate_query_embedding(query)

        results: ChromaQueryResult = self.collection.query(  # type: ignore[assignment]
            query_embeddings=cast(Any, [query_embedding]),
            n_results=top_k_value,
            include=["documents", "metadatas", "distances"],
        )

        return self._format_retrieval_results(results)

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the collection"""
        count: int = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": Config.CHROMA_PERSIST_DIR,
        }

    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.

        Deletes the collection and recreates it, handling errors gracefully.
        """
        self._delete_collection_safe()
        self.collection = self._get_or_create_collection()
        logger.info(f"Cleared and recreated collection: {self.collection_name}")

    def format_retrieved_context(self, chunks: list[dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string for LLM.

        Args:
            chunks: Retrieved chunks

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."

        context_parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            metadata: dict[str, Any] = chunk.get("metadata", {})
            source: str = metadata.get("title", metadata.get("url", "Unknown source"))
            text: str = chunk["text"]

            context_parts.append(f"[Source {i}: {source}]\n{text}\n")

        return "\n---\n".join(context_parts)


def create_rag_pipeline(collection_name: str | None = None) -> RAGPipeline:
    """
    Factory function to create RAG pipeline instance.

    Args:
        collection_name: Optional collection name

    Returns:
        Configured RAGPipeline instance
    """
    return RAGPipeline(collection_name=collection_name)
