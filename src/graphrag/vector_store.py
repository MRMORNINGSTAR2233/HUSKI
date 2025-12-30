"""ChromaDB vector store for document retrieval."""

import logging
import time
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import ChromaConfig, LLMConfig
from .models import VectorResult

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector embeddings and semantic search using ChromaDB."""

    def __init__(self, chroma_config: ChromaConfig, llm_config: LLMConfig):
        """Initialize vector store.

        Args:
            chroma_config: ChromaDB configuration
            llm_config: LLM configuration for embeddings
        """
        self.chroma_config = chroma_config
        self.llm_config = llm_config
        self._client: Optional[chromadb.Client] = None
        self._collection = None

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def connect(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Create persistent client
            self._client = chromadb.Client(
                Settings(
                    persist_directory=self.chroma_config.persist_directory,
                    anonymized_telemetry=False,
                )
            )

            # Set up OpenAI embedding function
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.llm_config.api_key,
                model_name=self.llm_config.embedding_model,
            )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.chroma_config.collection_name,
                embedding_function=openai_ef,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                f"Connected to ChromaDB collection: {self.chroma_config.collection_name}"
            )
            logger.info(f"Collection size: {self._collection.count()} documents")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        chunk: bool = True,
    ) -> int:
        """Add documents to vector store.

        Args:
            documents: List of text documents
            metadatas: Optional metadata for each document
            ids: Optional document IDs
            chunk: Whether to chunk documents before adding

        Returns:
            Number of chunks/documents added
        """
        if not self._collection:
            raise RuntimeError("Not connected to ChromaDB. Call connect() first.")

        processed_docs = []
        processed_metadata = []
        processed_ids = []

        for i, doc in enumerate(documents):
            if chunk:
                # Split document into chunks
                chunks = self.text_splitter.split_text(doc)
                for j, chunk_text in enumerate(chunks):
                    processed_docs.append(chunk_text)

                    # Copy metadata and add chunk info
                    meta = metadatas[i].copy() if metadatas else {}
                    meta.update({"chunk_index": j, "total_chunks": len(chunks)})
                    processed_metadata.append(meta)

                    # Generate ID
                    doc_id = ids[i] if ids else f"doc_{i}"
                    processed_ids.append(f"{doc_id}_chunk_{j}")
            else:
                processed_docs.append(doc)
                processed_metadata.append(metadatas[i] if metadatas else {})
                processed_ids.append(ids[i] if ids else f"doc_{i}")

        try:
            self._collection.add(
                documents=processed_docs,
                metadatas=processed_metadata,
                ids=processed_ids,
            )
            logger.info(f"Added {len(processed_docs)} chunks to vector store")
            return len(processed_docs)

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> VectorResult:
        """Perform semantic search.

        Args:
            query: Search query
            top_k: Number of results to return
            where: Metadata filters
            where_document: Document content filters

        Returns:
            VectorResult with search results
        """
        if not self._collection:
            raise RuntimeError("Not connected to ChromaDB. Call connect() first.")

        start_time = time.time()

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                where_document=where_document,
            )

            execution_time = (time.time() - start_time) * 1000

            # Extract results
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            logger.info(
                f"Vector search completed in {execution_time:.2f}ms, "
                f"found {len(documents)} results"
            )

            return VectorResult(
                documents=documents,
                metadatas=metadatas,
                distances=distances,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise

    def delete_documents(self, ids: Optional[List[str]] = None, where: Optional[Dict[str, Any]] = None) -> None:
        """Delete documents from vector store.

        Args:
            ids: Document IDs to delete
            where: Metadata filter for deletion
        """
        if not self._collection:
            raise RuntimeError("Not connected to ChromaDB. Call connect() first.")

        try:
            if ids:
                self._collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} documents")
            elif where:
                self._collection.delete(where=where)
                logger.info(f"Deleted documents matching filter: {where}")
            else:
                logger.warning("No deletion criteria provided")

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dictionary with collection stats
        """
        if not self._collection:
            raise RuntimeError("Not connected to ChromaDB. Call connect() first.")

        return {
            "name": self._collection.name,
            "count": self._collection.count(),
            "metadata": self._collection.metadata,
        }

    def reset_collection(self) -> None:
        """Delete and recreate collection (WARNING: destroys all data)."""
        if not self._client:
            raise RuntimeError("Not connected to ChromaDB. Call connect() first.")

        try:
            self._client.delete_collection(name=self.chroma_config.collection_name)
            logger.warning(f"Deleted collection: {self.chroma_config.collection_name}")
            self.connect()  # Recreate collection

        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # ChromaDB client doesn't need explicit closing
        pass
