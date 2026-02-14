"""Configuration management for the RAG pipeline"""

import os
from typing import Optional
from dotenv import load_dotenv


class Config:
    """Configuration class for RAG pipeline settings"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration from environment variables
        
        Args:
            env_file: Path to .env file. If None, uses default .env
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # Model settings
        self.model_name = os.getenv("MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Vector store settings
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH", "./chroma_db")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        
        # Generation parameters
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "256"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("TOP_P", "0.9"))
    
    def __repr__(self):
        return (
            f"Config(model_name={self.model_name}, "
            f"embedding_model={self.embedding_model}, "
            f"vector_store_path={self.vector_store_path})"
        )
