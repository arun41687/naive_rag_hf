"""Main RAG pipeline orchestrator"""

from typing import List, Optional, Tuple
from .config import Config
from .document_loader import DocumentLoader
from .vector_store import VectorStoreManager
from .llm_wrapper import LLMWrapper


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline using HuggingFace models
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize RAG pipeline
        
        Args:
            config: Configuration object. If None, uses default config
        """
        self.config = config or Config()
        
        # Initialize components
        print("Initializing RAG Pipeline...")
        
        self.document_loader = DocumentLoader(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self.vector_store = VectorStoreManager(
            embedding_model=self.config.embedding_model,
            persist_directory=self.config.vector_store_path
        )
        
        self.llm = LLMWrapper(
            model_name=self.config.model_name,
            token=self.config.huggingface_token,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )
        
        print("RAG Pipeline initialized successfully!")
    
    def add_documents(self, documents: List[str]):
        """
        Add documents to the knowledge base
        
        Args:
            documents: List of document texts to add
        """
        all_chunks = []
        for doc in documents:
            chunks = self.document_loader.load_and_chunk_text(doc)
            all_chunks.extend(chunks)
        
        print(f"Adding {len(all_chunks)} chunks to vector store...")
        self.vector_store.add_documents(all_chunks)
        print(f"Total documents in store: {self.vector_store.count()}")
    
    def add_document_files(self, file_paths: List[str]):
        """
        Add documents from files to the knowledge base
        
        Args:
            file_paths: List of file paths to add
        """
        all_chunks = []
        for file_path in file_paths:
            chunks = self.document_loader.load_and_chunk_file(file_path)
            all_chunks.extend(chunks)
        
        print(f"Adding {len(all_chunks)} chunks from {len(file_paths)} files to vector store...")
        self.vector_store.add_documents(all_chunks)
        print(f"Total documents in store: {self.vector_store.count()}")
    
    def retrieve(self, query: str, n_results: int = 3) -> Tuple[List[str], List[float]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query text
            n_results: Number of results to retrieve
            
        Returns:
            Tuple of (documents, distances)
        """
        return self.vector_store.query(query, n_results=n_results)
    
    def generate_prompt(self, query: str, context_docs: List[str]) -> str:
        """
        Generate a prompt combining context and query
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            
        Returns:
            Formatted prompt
        """
        context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])
        
        prompt = f"""You are a helpful assistant. Use the following context to answer the question. If you cannot answer based on the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def query(self, question: str, n_results: int = 3, return_context: bool = False) -> str:
        """
        Query the RAG pipeline
        
        Args:
            question: Question to answer
            n_results: Number of context documents to retrieve
            return_context: Whether to return retrieved context along with answer
            
        Returns:
            Generated answer (or tuple of answer and context if return_context=True)
        """
        # Retrieve relevant documents
        context_docs, distances = self.retrieve(question, n_results=n_results)
        
        if not context_docs:
            return "No relevant documents found in the knowledge base."
        
        # Generate prompt
        prompt = self.generate_prompt(question, context_docs)
        
        # Generate answer
        answer = self.llm.generate(prompt)
        
        if return_context:
            return answer, context_docs
        
        return answer
    
    def clear_knowledge_base(self):
        """Clear all documents from the knowledge base"""
        self.vector_store.clear()
        print("Knowledge base cleared.")
