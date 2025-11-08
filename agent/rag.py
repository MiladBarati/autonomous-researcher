"""
RAG Pipeline for Document Storage and Retrieval

Implements chunking, embedding, vector storage, and semantic retrieval
using ChromaDB and sentence-transformers.
"""

import hashlib
from datetime import datetime
from typing import Any

import chromadb
from chromadb import Collection
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]

from agent.logger import get_logger
from agent.validation import ValidationError, validate_query
from config import Config

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
        self.client: Any = chromadb.PersistentClient(
            path=Config.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Create or get collection
        self.collection: Collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one"""
        try:
            collection: Any = self.client.get_collection(name=self.collection_name)
            logger.debug(f"Using existing collection: {self.collection_name}")
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")

        return collection  # type: ignore[no-any-return]

    def chunk_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents with 'content' field

        Returns:
            List of chunk dictionaries with metadata
        """
        chunks: list[dict[str, Any]] = []

        for doc in documents:
            content: str = doc.get("content", "")
            if not content:
                continue

            # Split into chunks
            text_chunks: list[str] = self.text_splitter.split_text(content)

            # Create chunk metadata
            for i, chunk_text in enumerate(text_chunks):
                chunk_id: str = hashlib.md5(
                    f"{doc.get('url', '')}{i}{chunk_text[:100]}".encode()
                ).hexdigest()

                chunk: dict[str, Any] = {
                    "id": chunk_id,
                    "text": chunk_text,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "source": doc.get("source", "unknown"),
                    "url": doc.get("url", ""),
                    "title": doc.get("title", ""),
                    "timestamp": datetime.now().isoformat(),
                }

                # Add any additional metadata from document
                for key in ["authors", "published", "categories"]:
                    if key in doc:
                        chunk[key] = str(doc[key])

                chunks.append(chunk)

        return chunks

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[list[float]]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of embedding vectors
        """
        texts: list[str] = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings.tolist()  # type: ignore[no-any-return]

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

        # Chunk documents
        logger.debug(f"Chunking {len(documents)} documents...")
        chunks: list[dict[str, Any]] = self.chunk_documents(documents)

        if not chunks:
            return 0

        logger.debug(f"Created {len(chunks)} chunks")

        # Generate embeddings
        logger.debug("Generating embeddings...")
        embeddings: list[list[float]] = self.embed_chunks(chunks)

        # Prepare data for ChromaDB
        ids: list[str] = [chunk["id"] for chunk in chunks]
        texts: list[str] = [chunk["text"] for chunk in chunks]
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            metadata: dict[str, Any] = {k: v for k, v in chunk.items() if k not in ["id", "text"]}
            # ChromaDB requires all metadata values to be strings, ints, floats, or bools
            metadata = {
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in metadata.items()
            }
            metadatas.append(metadata)

        # Store in ChromaDB
        logger.debug("Storing in ChromaDB...")
        # Convert embeddings and metadatas to compatible types
        embeddings_any: Any = embeddings
        metadatas_any: Any = metadatas
        self.collection.add(
            ids=ids, embeddings=embeddings_any, documents=texts, metadatas=metadatas_any
        )

        logger.info(f"Successfully stored {len(chunks)} chunks")
        return len(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with metadata
        """
        # Validate and sanitize query
        try:
            query = validate_query(query)
        except ValidationError as e:
            logger.error(f"Invalid retrieval query: {e}")
            return []

        top_k_value: int = top_k or Config.TOP_K_RESULTS

        # Generate query embedding
        query_embedding: list[float] = self.embedding_model.encode(
            [query], show_progress_bar=False, convert_to_numpy=True
        ).tolist()[0]

        # Query ChromaDB
        query_embeddings_any: Any = [query_embedding]
        results: Any = self.collection.query(
            query_embeddings=query_embeddings_any,
            n_results=top_k_value,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        retrieved_chunks: list[dict[str, Any]] = []
        if results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                chunk: dict[str, Any] = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity_score": 1
                    - results["distances"][0][i],  # Convert distance to similarity
                    "rank": i + 1,
                }
                retrieved_chunks.append(chunk)

        return retrieved_chunks

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the collection"""
        count: int = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": Config.CHROMA_PERSIST_DIR,
        }

    def clear_collection(self) -> None:
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._get_or_create_collection()
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}", exc_info=True)

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
