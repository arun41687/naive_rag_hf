"""
Naive RAG HuggingFace Package
A simple RAG pipeline using HuggingFace's Microsoft/Phi3-mini-4k-instruct model
"""

__version__ = "0.1.0"
__author__ = "arun41687"

from .rag_pipeline import RAGPipeline
from .document_loader import DocumentLoader
from .vector_store import VectorStoreManager
from .llm_wrapper import LLMWrapper

__all__ = ["RAGPipeline", "DocumentLoader", "VectorStoreManager", "LLMWrapper"]
