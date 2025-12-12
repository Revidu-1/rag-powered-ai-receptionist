"""
Interactive RAG Query Interface
Simple script to ask questions and get answers from the RAG system.
"""

from mini_rag import MiniRAG
import sys


def main():
    """Interactive query interface for the RAG system."""
    
    print("="*80)
    print("MINI-RAG QUERY INTERFACE")
    print("="*80)
    print("\nInitializing RAG system...")
    print("(This may take a moment to load the embedding model)\n")
    
    # Initialize RAG system
    use_reranking = input("Enable reranking? (y/n, default=n): ").strip().lower() == 'y'
    rag = MiniRAG(use_reranking=use_reranking)
    
    # Load documents
    file_paths = ['handbook.txt', 'faq.txt', 'blog.txt']
    print(f"\nLoading documents: {', '.join(file_paths)}...")
    rag.load_documents(file_paths)
    
    # Generate embeddings
    print("\nGenerating embeddings and building index...")
    rag.generate_embeddings()
    
    print("\n" + "="*80)
    print("READY! Ask questions about the documents.")
    print("Type 'quit' or 'exit' to stop.")
    print("="*80 + "\n")
    
    while True:
        try:
            # Get user question
            question = input("Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            # Get number of chunks to retrieve
            try:
                k = input("Number of chunks to retrieve (default=3): ").strip()
                k = int(k) if k else 3
            except ValueError:
                k = 3
            
            print("\n" + "-"*80)
            print(f"Query: {question}")
            print("-"*80)
            
            # Get answer
            answer, retrieved = rag.answer(question, k=k)
            
            # Display answer
            print(f"\nAnswer:\n{answer}\n")
            
            # Display retrieved chunks
            print(f"Retrieved {len(retrieved)} chunks:")
            for i, chunk in enumerate(retrieved, 1):
                print(f"\n  Chunk {i}:")
                print(f"    Source: {chunk['metadata']['filename']}")
                print(f"    Relevance Score: {chunk['score']:.4f}")
                print(f"    Preview: {chunk['text'][:150]}...")
            
            print("\n" + "="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            continue


if __name__ == "__main__":
    main()
