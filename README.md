# Mini-RAG System

A simple Retrieval-Augmented Generation (RAG) system that demonstrates the core concepts of document retrieval and question answering.

## Features

- **Document Loading**: Loads multiple text files (handbook, FAQ, blog)
- **Smart Chunking**: Splits text into chunks of ~300-500 tokens with overlap
- **Embedding Generation**: Uses sentence transformers for semantic embeddings
- **FAISS Vector Store**: Fast similarity search using Facebook AI Similarity Search
- **LLM Integration**: Answers questions using retrieved context (requires OpenAI API key)
- **Reranking Toggle**: Optional reranking mode for improved retrieval quality

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Set OpenAI API key for LLM features:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

If you don't set the API key, the system will still work but will return concatenated retrieved chunks instead of LLM-generated answers.

## Usage

### Interactive Query Interface (Recommended)

The easiest way to use the RAG system is with the interactive query interface:

```bash
python query_rag.py
```

This will:
1. Load all documents and build the index
2. Let you ask questions interactively
3. Show answers and retrieved chunks
4. Allow you to adjust the number of chunks retrieved

Example session:
```
Question: How many days of PTO do employees get?
Number of chunks to retrieve (default=3): 3

Answer: [Retrieved answer from documents]

Retrieved 3 chunks:
  Chunk 1: Source: handbook.txt, Score: 0.5621
  ...
```

### Programmatic Usage

You can also use the RAG system in your own Python code:

```python
from mini_rag import MiniRAG

# Initialize
rag = MiniRAG(use_reranking=False)  # Set to True for reranking

# Load documents
rag.load_documents(['handbook.txt', 'faq.txt', 'blog.txt'])

# Generate embeddings and build index
rag.generate_embeddings()

# Ask a question
answer, retrieved_chunks = rag.answer("What is the PTO policy?", k=3)

print(answer)
for chunk in retrieved_chunks:
    print(f"Source: {chunk['metadata']['filename']}, Score: {chunk['score']}")
```

### Demo Script

Run the main demo with predefined questions:
```bash
python mini_rag.py
```

### Testing

Run comprehensive tests with 5 questions:
```bash
python test_rag.py
```

This will:
- Test the system with 5 different questions
- Compare results with and without reranking
- Evaluate where the system works vs. fails
- Provide recommendations for improvement

## How It Works

1. **Loading**: Documents are loaded from text files
2. **Chunking**: Text is split into semantic chunks of ~300-500 tokens
3. **Embedding**: Each chunk is converted to a vector embedding
4. **Indexing**: Embeddings are stored in a FAISS index for fast retrieval
5. **Retrieval**: For a query, the top-k most relevant chunks are retrieved
6. **Answering**: An LLM (or simple concatenation) generates an answer from retrieved context

## Files

- `mini_rag.py`: Main RAG implementation
- `test_rag.py`: Test script with 5 questions
- `handbook.txt`: Sample employee handbook
- `faq.txt`: Sample FAQ document
- `blog.txt`: Sample blog posts

## Customization

You can modify the chunking strategy, embedding model, or retrieval parameters in `mini_rag.py`:

```python
# Change chunk size
rag = MiniRAG()
rag.chunker.token_limit = 500  # Adjust token limit

# Enable reranking
rag = MiniRAG(use_reranking=True)

# Use different embedding model
rag = MiniRAG(embedding_model="all-mpnet-base-v2")  # Larger, better model
```

## Notes

- The system works without an OpenAI API key (uses simple concatenation)
- Reranking can improve results but adds computation time
- Chunk size of 300-500 tokens works well for most use cases
- FAISS uses cosine similarity for retrieval
