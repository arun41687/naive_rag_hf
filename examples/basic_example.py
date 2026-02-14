"""
Basic example demonstrating the RAG pipeline
"""

from naive_rag import RAGPipeline

def main():
    # Initialize the RAG pipeline
    print("=" * 60)
    print("RAG Pipeline Basic Example")
    print("=" * 60)
    
    pipeline = RAGPipeline()
    
    # Sample documents to add to knowledge base
    documents = [
        """
        Python is a high-level, interpreted programming language known for its 
        simplicity and readability. It was created by Guido van Rossum and first 
        released in 1991. Python supports multiple programming paradigms including 
        procedural, object-oriented, and functional programming.
        """,
        """
        Machine learning is a subset of artificial intelligence that focuses on 
        developing systems that can learn from and make decisions based on data. 
        Popular machine learning frameworks include TensorFlow, PyTorch, and 
        scikit-learn. Python is the most widely used language for machine learning.
        """,
        """
        The HuggingFace Transformers library provides thousands of pretrained models 
        for natural language processing tasks. It supports models like BERT, GPT, 
        T5, and many others. The library makes it easy to use state-of-the-art 
        models in your applications.
        """
    ]
    
    # Add documents to the knowledge base
    print("\nAdding documents to knowledge base...")
    pipeline.add_documents(documents)
    
    # Query the pipeline
    print("\n" + "=" * 60)
    print("Asking questions...")
    print("=" * 60)
    
    questions = [
        "What is Python?",
        "What programming language is most used for machine learning?",
        "What does HuggingFace provide?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 60)
        
        # Get answer with context
        answer, context = pipeline.query(question, n_results=2, return_context=True)
        
        print(f"Answer: {answer}")
        print(f"\nRetrieved context snippets: {len(context)}")
        
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
