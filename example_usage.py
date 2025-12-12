"""
Example: How to use the Mini-RAG system programmatically
"""

from mini_rag import MiniRAG


# Example 1: Basic usage
def example_basic():
    """Basic example of using the RAG system."""
    print("="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    # Initialize RAG system
    rag = MiniRAG(use_reranking=False)
    
    # Load documents
    rag.load_documents(['handbook.txt', 'faq.txt', 'blog.txt'])
    
    # Generate embeddings and build index
    rag.generate_embeddings()
    
    # Ask a question
    question = "How many days of PTO do employees get per year?"
    answer, retrieved = rag.answer(question, k=3)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")
    print(f"\nRetrieved {len(retrieved)} chunks:")
    for i, chunk in enumerate(retrieved, 1):
        print(f"  {i}. {chunk['metadata']['filename']} (score: {chunk['score']:.4f})")


# Example 2: With reranking
def example_with_reranking():
    """Example using reranking for better results."""
    print("\n" + "="*80)
    print("EXAMPLE 2: With Reranking")
    print("="*80)
    
    rag = MiniRAG(use_reranking=True)
    rag.load_documents(['handbook.txt', 'faq.txt', 'blog.txt'])
    rag.generate_embeddings()
    
    question = "What mental health resources are available?"
    answer, retrieved = rag.answer(question, k=3)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer[:200]}...")  # First 200 chars
    print(f"\nTop chunk: {retrieved[0]['metadata']['filename']} (score: {retrieved[0]['score']:.4f})")


# Example 3: Just retrieval (no LLM)
def example_retrieval_only():
    """Example of just retrieving relevant chunks without generating an answer."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Retrieval Only")
    print("="*80)
    
    rag = MiniRAG()
    rag.load_documents(['handbook.txt', 'faq.txt', 'blog.txt'])
    rag.generate_embeddings()
    
    question = "What are the working hours?"
    retrieved = rag.retrieve(question, k=2)
    
    print(f"\nQuestion: {question}")
    print(f"\nTop {len(retrieved)} most relevant chunks:")
    for i, chunk in enumerate(retrieved, 1):
        print(f"\n  Chunk {i}:")
        print(f"    Source: {chunk['metadata']['filename']}")
        print(f"    Score: {chunk['score']:.4f}")
        print(f"    Text: {chunk['text'][:200]}...")


if __name__ == "__main__":
    example_basic()
    example_with_reranking()
    example_retrieval_only()
    
    print("\n" + "="*80)
    print("For interactive use, run: python query_rag.py")
    print("="*80)
