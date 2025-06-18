# ===============================
# 4. cadgen_project/config/rag_config.py
# ===============================

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from .settings import settings


class RAGConfig(BaseModel):
    """RAG system configuration."""

    # Embedding Configuration
    embedding_model: str = Field(default_factory=lambda: settings.embedding_model)
    embedding_dimension: int = Field(default=384)

    # Chunking Configuration
    chunk_size: int = Field(default_factory=lambda: settings.chunk_size)
    chunk_overlap: int = Field(default_factory=lambda: settings.chunk_overlap)

    # Retrieval Configuration
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # ChromaDB Configuration
    persist_directory: str = Field(
        default_factory=lambda: settings.chroma_persist_directory
    )
    collection_name: str = Field(default="cadquery_docs")

    # Document Processing
    supported_formats: List[str] = Field(
        default=[".md", ".txt", ".pdf", ".html", ".py"]
    )
    max_file_size_mb: int = Field(default=10)

    def get_text_splitter_config(self) -> Dict[str, Any]:
        """Get configuration for text splitting."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "length_function": len,
            "separators": ["\n\n", "\n", " ", ""],
        }

    def get_chroma_config(self) -> Dict[str, Any]:
        """Get ChromaDB configuration."""
        return {
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name,
        }


# Global RAG config instance
rag_config = RAGConfig()
