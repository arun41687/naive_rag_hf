"""
Advanced example with custom configuration and file loading
"""

import os
from naive_rag import RAGPipeline
from naive_rag.config import Config

def main():
    print("=" * 60)
    print("RAG Pipeline Advanced Example")
    print("=" * 60)
    
    # Create custom configuration
    config = Config()
    config.max_new_tokens = 512  # Generate longer responses
    config.temperature = 0.5     # More deterministic responses
    
    print(f"\nUsing configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Embeddings: {config.embedding_model}")
    print(f"  Chunk size: {config.chunk_size}")
    print(f"  Max tokens: {config.max_new_tokens}")
    
    # Initialize pipeline with custom config
    pipeline = RAGPipeline(config)
    
    # Create a sample document file
    sample_file = "/tmp/sample_document.txt"
    with open(sample_file, 'w') as f:
        f.write("""
        Retrieval-Augmented Generation (RAG) is a technique that combines 
        information retrieval with text generation. It works by first retrieving 
        relevant documents from a knowledge base, then using those documents as 
        context for a language model to generate responses.
        
        The RAG approach has several advantages:
        1. It grounds the model's responses in factual information
        2. It allows updating the knowledge base without retraining the model
        3. It can provide citations and source information
        4. It reduces hallucinations by constraining generation to retrieved context
        
        RAG systems typically consist of three main components:
        - A document store or vector database
        - An embedding model for semantic search
        - A language model for generation
        """)
    
    print(f"\nCreated sample document: {sample_file}")
    
    # Add document from file
    print("\nLoading document from file...")
    pipeline.add_document_files([sample_file])
    
    # Interactive Q&A
    print("\n" + "=" * 60)
    print("Interactive Q&A Session")
    print("=" * 60)
    
    questions = [
        "What is RAG?",
        "What are the advantages of RAG?",
        "What components does a RAG system have?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        print("-" * 60)
        
        answer = pipeline.query(question, n_results=3)
        print(f"A: {answer}")
    
    # Cleanup
    os.remove(sample_file)
    
    print("\n" + "=" * 60)
    print("Advanced example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
