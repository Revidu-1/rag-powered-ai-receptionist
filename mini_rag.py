"""
Mini-RAG System
A simple Retrieval-Augmented Generation implementation with:
- Text chunking (~300-500 tokens)
- Embedding generation
- FAISS vector store
- LLM-based question answering
"""

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import tiktoken
try:
    # Try new langchain import first
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
    except ImportError:
        # Fallback to old import
        from langchain.chat_models import ChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain not available. LLM features will be limited.")


class TextChunker:
    """Handles text splitting into chunks of approximately 300-500 tokens."""
    
    def __init__(self, token_limit: int = 400, overlap: int = 50):
        self.token_limit = token_limit
        self.overlap = overlap
        # Use tiktoken for token counting (approximates OpenAI tokenization)
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks based on token count.
        Returns list of dictionaries with 'text' and 'metadata' keys.
        """
        # Tokenize the entire text
        tokens = self.encoding.encode(text)
        chunks = []
        
        # Split into paragraphs first for better semantic boundaries
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = len(self.encoding.encode(para))
            
            # If a single paragraph exceeds limit, split it by sentences
            if para_tokens > self.token_limit:
                # First, save current chunk if it has content
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'metadata': metadata or {}
                    })
                    current_chunk = []
                    current_tokens = 0
                
                # Split large paragraph by sentences
                sentences = para.split('. ')
                for i, sentence in enumerate(sentences):
                    if i < len(sentences) - 1:
                        sentence += '. '
                    sent_tokens = len(self.encoding.encode(sentence))
                    
                    if current_tokens + sent_tokens > self.token_limit and current_chunk:
                        chunk_text = '\n\n'.join(current_chunk)
                        chunks.append({
                            'text': chunk_text,
                            'metadata': metadata or {}
                        })
                        # Keep some overlap
                        overlap_text = '\n\n'.join(current_chunk[-self.overlap//10:]) if len(current_chunk) > self.overlap//10 else ''
                        current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                        current_tokens = len(self.encoding.encode('\n\n'.join(current_chunk)))
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sent_tokens
            else:
                # Check if adding this paragraph would exceed limit
                if current_tokens + para_tokens > self.token_limit and current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'metadata': metadata or {}
                    })
                    # Keep overlap
                    overlap_text = '\n\n'.join(current_chunk[-2:]) if len(current_chunk) > 2 else current_chunk[0]
                    current_chunk = [overlap_text, para] if len(current_chunk) > 1 else [para]
                    current_tokens = len(self.encoding.encode('\n\n'.join(current_chunk)))
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'metadata': metadata or {}
            })
        
        return chunks


class MiniRAG:
    """Main RAG system with embedding generation and FAISS storage."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", use_reranking: bool = False):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: Name of the sentence transformer model
            use_reranking: Whether to use reranking for retrieved chunks
        """
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunker = TextChunker(token_limit=400, overlap=50)
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.use_reranking = use_reranking
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
    
    def load_documents(self, file_paths: List[str]) -> None:
        """Load and chunk documents from file paths."""
        all_chunks = []
        
        for file_path in file_paths:
            print(f"Loading {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            metadata = {
                'source': file_path,
                'filename': os.path.basename(file_path)
            }
            
            chunks = self.chunker.chunk_text(text, metadata)
            print(f"  Split into {len(chunks)} chunks")
            all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        print(f"Total chunks: {len(self.chunks)}")
    
    def generate_embeddings(self) -> None:
        """Generate embeddings for all chunks and build FAISS index."""
        if not self.chunks:
            raise ValueError("No chunks available. Load documents first.")
        
        print("Generating embeddings...")
        texts = [chunk['text'] for chunk in self.chunks]
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        print("Building FAISS index...")
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_normalized = self.embeddings / norms
        
        # Create FAISS index (using inner product for normalized vectors = cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings_normalized.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Args:
            query: The search query
            k: Number of chunks to retrieve
            
        Returns:
            List of chunk dictionaries with relevance scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call generate_embeddings() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding_normalized = query_embedding / query_norm
        
        # Search FAISS index
        k = min(k, len(self.chunks))
        scores, indices = self.index.search(query_embedding_normalized.astype('float32'), k)
        
        # Retrieve chunks with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(score)
            results.append(chunk)
        
        # Optional reranking (simple approach: re-score with query similarity)
        if self.use_reranking and len(results) > 1:
            results = self._rerank(query, results)
        
        return results
    
    def _rerank(self, query: str, results: List[Dict], top_k: int = None) -> List[Dict]:
        """Simple reranking by recalculating similarity scores."""
        if top_k is None:
            top_k = len(results)
        
        # Re-encode query and chunk texts for more accurate scoring
        query_emb = self.embedding_model.encode([query])[0]
        query_norm = np.linalg.norm(query_emb)
        
        for result in results:
            chunk_emb = self.embedding_model.encode([result['text']])[0]
            chunk_norm = np.linalg.norm(chunk_emb)
            # Cosine similarity
            similarity = np.dot(query_emb, chunk_emb) / (query_norm * chunk_norm)
            result['score'] = float(similarity)
        
        # Sort by new scores
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def answer(self, query: str, k: int = 3, use_llm: bool = True) -> Tuple[str, List[Dict]]:
        """
        Retrieve relevant chunks and generate an answer.
        
        Args:
            query: The question to answer
            k: Number of chunks to retrieve
            use_llm: Whether to use LLM for answer generation (if False, returns retrieved chunks)
            
        Returns:
            Tuple of (answer, retrieved_chunks)
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(query, k=k)
        
        if not use_llm:
            return "Retrieved chunks only (LLM disabled)", retrieved_chunks
        
        # Check if OpenAI API key is available
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or not LANGCHAIN_AVAILABLE:
            print("Warning: OPENAI_API_KEY not set or langchain not available. Using simple concatenation of retrieved chunks.")
            answer = "\n\n".join([f"From {chunk['metadata']['filename']}:\n{chunk['text']}" 
                                 for chunk in retrieved_chunks])
            return answer, retrieved_chunks
        
        # Generate answer using LLM
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            
            # Build context from retrieved chunks
            context = "\n\n---\n\n".join([
                f"Source: {chunk['metadata']['filename']}\n{chunk['text']}"
                for chunk in retrieved_chunks
            ])
            
            messages = [
                SystemMessage(content="You are a helpful assistant that answers questions based on the provided context. If the answer cannot be found in the context, say so clearly."),
                HumanMessage(content=f"""Context information:
{context}

Question: {query}

Based on the context above, please answer the question. If the answer is not in the context, say "I cannot find this information in the provided documents." """)
            ]
            
            response = llm(messages)
            answer = response.content
            
        except Exception as e:
            print(f"Error using LLM: {e}")
            # Fallback to simple concatenation
            answer = "\n\n".join([f"From {chunk['metadata']['filename']}:\n{chunk['text']}" 
                                 for chunk in retrieved_chunks])
        
        return answer, retrieved_chunks


def main():
    """Main function to demonstrate the RAG system."""
    # Initialize RAG system
    rag = MiniRAG(use_reranking=False)
    
    # Load documents
    file_paths = ['handbook.txt', 'faq.txt', 'blog.txt']
    rag.load_documents(file_paths)
    
    # Generate embeddings and build index
    rag.generate_embeddings()
    
    # Test queries
    test_questions = [
        "What is the PTO policy?",
        "How do I request time off?",
        "What are the working hours?",
        "How do I access the company VPN?",
        "What mental health resources are available?"
    ]
    
    print("\n" + "="*80)
    print("TESTING RAG SYSTEM")
    print("="*80 + "\n")
    
    results_summary = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}: {question}")
        print("-" * 80)
        
        answer, retrieved = rag.answer(question, k=3)
        
        print(f"Answer:\n{answer}\n")
        print(f"Retrieved {len(retrieved)} chunks:")
        for j, chunk in enumerate(retrieved, 1):
            print(f"  {j}. [{chunk['metadata']['filename']}] Score: {chunk['score']:.4f}")
            print(f"     Preview: {chunk['text'][:100]}...")
        print("\n" + "="*80 + "\n")
        
        results_summary.append({
            'question': question,
            'answer_length': len(answer),
            'chunks_retrieved': len(retrieved),
            'top_score': retrieved[0]['score'] if retrieved else 0
        })
    
    # Summary
    print("\nSUMMARY")
    print("="*80)
    for result in results_summary:
        print(f"Q: {result['question']}")
        print(f"  - Answer length: {result['answer_length']} chars")
        print(f"  - Chunks retrieved: {result['chunks_retrieved']}")
        print(f"  - Top relevance score: {result['top_score']:.4f}")
        print()


if __name__ == "__main__":
    main()
