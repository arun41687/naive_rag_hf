# Naive RAG with HuggingFace

A simple and effective Retrieval-Augmented Generation (RAG) pipeline using HuggingFace's Microsoft Phi-3-mini-4k-instruct model.

## Overview

This project implements a naive RAG pipeline that combines document retrieval with language generation to provide contextually relevant answers. The pipeline uses:

- **LLM**: Microsoft Phi-3-mini-4k-instruct (via HuggingFace Transformers)
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB for efficient similarity search
- **Document Processing**: Simple text chunking with overlap

## Features

- 🚀 Easy-to-use API for building RAG applications
- 📚 Support for adding documents from text or files
- 🔍 Semantic search using sentence embeddings
- 🤖 Integration with HuggingFace's Phi-3 model
- ⚙️ Configurable via environment variables or config objects
- 💾 Persistent vector storage with ChromaDB

## Installation

### Prerequisites

- Python 3.9 or higher
- (Optional) CUDA-capable GPU for faster inference

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or using the package:

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from naive_rag import RAGPipeline

# Initialize the pipeline
pipeline = RAGPipeline()

# Add documents to knowledge base
documents = [
    "Python is a programming language created by Guido van Rossum.",
    "Machine learning is a subset of artificial intelligence.",
    "HuggingFace provides pretrained transformer models."
]
pipeline.add_documents(documents)

# Query the pipeline
answer = pipeline.query("What is Python?")
print(answer)
```

### Advanced Usage

```python
from naive_rag import RAGPipeline
from naive_rag.config import Config

# Create custom configuration
config = Config()
config.max_new_tokens = 512
config.temperature = 0.5

# Initialize with custom config
pipeline = RAGPipeline(config)

# Add documents from files
pipeline.add_document_files(["document1.txt", "document2.txt"])

# Query with context
answer, context = pipeline.query(
    "What is RAG?",
    n_results=3,
    return_context=True
)
print(f"Answer: {answer}")
print(f"Context: {context}")
```

## Configuration

Configuration can be set via environment variables (create a `.env` file):

```bash
# Model configuration
MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional: HuggingFace token for private models
# HUGGINGFACE_TOKEN=your_token_here

# Vector store configuration
VECTOR_STORE_PATH=./chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Generation parameters
MAX_NEW_TOKENS=256
TEMPERATURE=0.7
TOP_P=0.9
```

## Examples

Run the provided examples:

```bash
# Basic example
python examples/basic_example.py

# Advanced example with file loading
python examples/advanced_example.py
```

## Project Structure

```
naive_rag_hf/
├── naive_rag/              # Main package
│   ├── __init__.py         # Package exports
│   ├── config.py           # Configuration management
│   ├── document_loader.py  # Document loading and chunking
│   ├── vector_store.py     # Vector store management
│   ├── llm_wrapper.py      # LLM wrapper for Phi-3
│   └── rag_pipeline.py     # Main RAG pipeline orchestrator
├── examples/               # Usage examples
│   ├── basic_example.py
│   └── advanced_example.py
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project metadata
├── .env.example           # Example environment variables
└── README.md              # This file
```

## API Reference

### RAGPipeline

Main class for the RAG pipeline.

**Methods:**

- `__init__(config: Optional[Config] = None)`: Initialize the pipeline
- `add_documents(documents: List[str])`: Add text documents to knowledge base
- `add_document_files(file_paths: List[str])`: Add documents from files
- `query(question: str, n_results: int = 3, return_context: bool = False)`: Query the pipeline
- `retrieve(query: str, n_results: int = 3)`: Retrieve relevant documents
- `clear_knowledge_base()`: Clear all documents

### Config

Configuration management class.

**Attributes:**

- `model_name`: LLM model identifier
- `embedding_model`: Embedding model identifier
- `vector_store_path`: Path to vector database
- `chunk_size`: Size of text chunks
- `chunk_overlap`: Overlap between chunks
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_p`: Nucleus sampling parameter

## How It Works

1. **Document Processing**: Documents are split into overlapping chunks
2. **Embedding**: Each chunk is converted to a vector embedding
3. **Storage**: Embeddings are stored in ChromaDB for fast retrieval
4. **Retrieval**: When queried, relevant chunks are retrieved using similarity search
5. **Generation**: Retrieved chunks are used as context for the LLM to generate an answer

## Performance Notes

- **CPU Mode**: Works but slower, suitable for testing and small-scale use
- **GPU Mode**: Recommended for production use, significantly faster inference
- **Memory**: Phi-3-mini-4k-instruct requires ~7-8GB GPU memory or 16GB+ RAM for CPU

## Limitations

- This is a "naive" implementation focused on simplicity
- No advanced features like re-ranking, query expansion, or hybrid search
- Basic text chunking without semantic awareness
- Single-turn conversations (no conversation history)

## Future Enhancements

- Support for multiple document formats (PDF, Word, HTML)
- Advanced chunking strategies
- Query rewriting and expansion
- Conversation memory
- Re-ranking of retrieved documents
- Hybrid search (keyword + semantic)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Acknowledgments

- [HuggingFace](https://huggingface.co/) for the Transformers library and Phi-3 model
- [Sentence-Transformers](https://www.sbert.net/) for embedding models
- [ChromaDB](https://www.trychroma.com/) for vector storage