"""Vector store manager for document embeddings and retrieval"""

import time
from typing import List, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np


class VectorStoreManager:
    """Manages vector storage and retrieval using ChromaDB"""
    
    def __init__(
        self, 
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents"
    ):
        """
        Initialize vector store manager
        
        Args:
            embedding_model: Name of the sentence-transformers model
            persist_directory: Directory to persist the vector database
            collection_name: Name of the collection in ChromaDB
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[str], ids: Optional[List[str]] = None):
        """
        Add documents to the vector store
        
        Args:
            documents: List of text documents to add
            ids: Optional list of IDs for the documents
        """
        if not documents:
            return
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, convert_to_numpy=True)
        
        # Generate IDs if not provided
        if ids is None:
            # Use timestamp in milliseconds + index to ensure uniqueness across multiple calls
            timestamp = int(time.time() * 1000)  # milliseconds
            ids = [f"doc_{timestamp}_{i}" for i in range(len(documents))]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=ids
        )
    
    def query(
        self, 
        query_text: str, 
        n_results: int = 3
    ) -> Tuple[List[str], List[float]]:
        """
        Query the vector store for similar documents
        
        Args:
            query_text: Query string
            n_results: Number of results to return
            
        Returns:
            Tuple of (documents, distances)
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text], convert_to_numpy=True)
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        return documents, distances
    
    def clear(self):
        """Clear all documents from the collection"""
        # Delete and recreate the collection
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def count(self) -> int:
        """
        Get the number of documents in the collection
        
        Returns:
            Number of documents
        """
        return self.collection.count()
