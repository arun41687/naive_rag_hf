"""Document loader for processing various document types"""

from typing import List, Optional
from pathlib import Path


class DocumentLoader:
    """Loads and preprocesses documents for the RAG pipeline"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document loader
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_text_file(self, file_path: str) -> str:
        """
        Load text from a file
        
        Args:
            file_path: Path to text file
            
        Returns:
            Text content of the file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_text(self, text: str) -> str:
        """
        Load text directly
        
        Args:
            text: Text content to load
            
        Returns:
            The text content
        """
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Check if this is the last chunk
            if end >= len(text):
                # Add final chunk (always include remaining text)
                final_chunk = text[start:]
                # If there are previous chunks and final chunk is very small,
                # merge it with the last chunk
                if chunks and len(final_chunk) < self.chunk_overlap:
                    chunks[-1] = chunks[-1] + final_chunk
                else:
                    chunks.append(final_chunk)
                break
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start forward by chunk_size minus overlap
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def load_and_chunk_file(self, file_path: str) -> List[str]:
        """
        Load a file and split it into chunks
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of text chunks
        """
        text = self.load_text_file(file_path)
        return self.chunk_text(text)
    
    def load_and_chunk_text(self, text: str) -> List[str]:
        """
        Load text and split it into chunks
        
        Args:
            text: Text to process
            
        Returns:
            List of text chunks
        """
        return self.chunk_text(text)
