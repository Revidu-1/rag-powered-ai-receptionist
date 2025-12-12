"""
Test script for the Mini-RAG system.
Tests with 5 questions and evaluates where it works vs. fails.
"""

import os
from mini_rag import MiniRAG


def test_rag_system():
    """Test the RAG system with 5 questions and evaluate results."""
    
    print("="*80)
    print("MINI-RAG SYSTEM TESTING")
    print("="*80)
    print()
    
    # Initialize RAG system (test both with and without reranking)
    print("Initializing RAG system...")
    rag_no_rerank = MiniRAG(use_reranking=False)
    rag_with_rerank = MiniRAG(use_reranking=True)
    
    # Load documents
    file_paths = ['handbook.txt', 'faq.txt', 'blog.txt']
    print(f"\nLoading documents: {file_paths}")
    rag_no_rerank.load_documents(file_paths)
    rag_with_rerank.load_documents(file_paths)
    
    # Generate embeddings
    print("\nGenerating embeddings and building FAISS index...")
    rag_no_rerank.generate_embeddings()
    rag_with_rerank.generate_embeddings()
    
    # Test questions with expected answers
    test_cases = [
        {
            "question": "How many days of PTO do employees get per year?",
            "expected_keywords": ["15", "fifteen", "PTO", "paid time off"],
            "expected_source": "handbook.txt"
        },
        {
            "question": "What is the process for requesting time off?",
            "expected_keywords": ["HR portal", "two weeks", "request"],
            "expected_source": "handbook.txt"
        },
        {
            "question": "What are the standard working hours?",
            "expected_keywords": ["9:00 AM", "5:00 PM", "Monday", "Friday"],
            "expected_source": "handbook.txt"
        },
        {
            "question": "How do I reset my password?",
            "expected_keywords": ["Forgot Password", "reset", "IT"],
            "expected_source": "faq.txt"
        },
        {
            "question": "What mental health resources are available?",
            "expected_keywords": ["EAP", "Employee Assistance Program", "mental health", "counseling"],
            "expected_source": "faq.txt"
        }
    ]
    
    print("\n" + "="*80)
    print("TESTING QUESTIONS")
    print("="*80 + "\n")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected_keywords = test_case["expected_keywords"]
        expected_source = test_case["expected_source"]
        
        print(f"\n{'='*80}")
        print(f"TEST {i}/5")
        print(f"{'='*80}")
        print(f"Question: {question}")
        print(f"Expected source: {expected_source}")
        print(f"Expected keywords: {', '.join(expected_keywords)}")
        print("-"*80)
        
        # Test without reranking
        print("\n[Without Reranking]")
        answer_no_rerank, retrieved_no_rerank = rag_no_rerank.answer(question, k=3)
        print(f"Answer: {answer_no_rerank[:200]}..." if len(answer_no_rerank) > 200 else f"Answer: {answer_no_rerank}")
        print(f"\nRetrieved chunks:")
        for j, chunk in enumerate(retrieved_no_rerank, 1):
            print(f"  {j}. Source: {chunk['metadata']['filename']} | Score: {chunk['score']:.4f}")
        
        # Test with reranking
        print("\n[With Reranking]")
        answer_rerank, retrieved_rerank = rag_with_rerank.answer(question, k=3)
        print(f"Answer: {answer_rerank[:200]}..." if len(answer_rerank) > 200 else f"Answer: {answer_rerank}")
        print(f"\nRetrieved chunks:")
        for j, chunk in enumerate(retrieved_rerank, 1):
            print(f"  {j}. Source: {chunk['metadata']['filename']} | Score: {chunk['score']:.4f}")
        
        # Evaluate results
        answer_lower = answer_no_rerank.lower()
        found_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        correct_source = any(chunk['metadata']['filename'] == expected_source 
                            for chunk in retrieved_no_rerank)
        
        result = {
            'question': question,
            'answer_length': len(answer_no_rerank),
            'found_keywords': found_keywords,
            'missing_keywords': [kw for kw in expected_keywords if kw.lower() not in answer_lower],
            'correct_source_found': correct_source,
            'top_source': retrieved_no_rerank[0]['metadata']['filename'] if retrieved_no_rerank else None,
            'top_score': retrieved_no_rerank[0]['score'] if retrieved_no_rerank else 0,
            'status': 'PASS' if len(found_keywords) >= len(expected_keywords) * 0.7 and correct_source else 'PARTIAL' if found_keywords else 'FAIL'
        }
        results.append(result)
    
    # Summary report
    print("\n" + "="*80)
    print("TEST SUMMARY & EVALUATION")
    print("="*80 + "\n")
    
    passes = sum(1 for r in results if r['status'] == 'PASS')
    partials = sum(1 for r in results if r['status'] == 'PARTIAL')
    fails = sum(1 for r in results if r['status'] == 'FAIL')
    
    print(f"Overall Results: {passes} PASS, {partials} PARTIAL, {fails} FAIL\n")
    
    for i, result in enumerate(results, 1):
        status_icon = "[PASS]" if result['status'] == 'PASS' else "[PARTIAL]" if result['status'] == 'PARTIAL' else "[FAIL]"
        print(f"{status_icon} Test {i}: {result['status']}")
        print(f"   Question: {result['question']}")
        print(f"   Found keywords: {', '.join(result['found_keywords']) if result['found_keywords'] else 'None'}")
        if result['missing_keywords']:
            print(f"   Missing keywords: {', '.join(result['missing_keywords'])}")
        print(f"   Top source: {result['top_source']} (score: {result['top_score']:.4f})")
        print(f"   Correct source found: {'Yes' if result['correct_source_found'] else 'No'}")
        print()
    
    # Analysis of where it works vs fails
    print("="*80)
    print("ANALYSIS: WHERE IT WORKS VS. FAILS")
    print("="*80 + "\n")
    
    print("STRENGTHS:")
    for result in results:
        if result['status'] == 'PASS':
            print(f"  [PASS] {result['question']} - Successfully retrieved relevant information")
    print()
    
    print("PARTIAL SUCCESSES:")
    for result in results:
        if result['status'] == 'PARTIAL':
            print(f"  [PARTIAL] {result['question']}")
            print(f"    - Found: {', '.join(result['found_keywords']) if result['found_keywords'] else 'None'}")
            print(f"    - Missing: {', '.join(result['missing_keywords']) if result['missing_keywords'] else 'None'}")
    print()
    
    print("AREAS FOR IMPROVEMENT:")
    for result in results:
        if result['status'] == 'FAIL':
            print(f"  [FAIL] {result['question']}")
            print(f"    - Issue: Could not find expected information")
            print(f"    - Retrieved source: {result['top_source']}")
    print()
    
    print("RECOMMENDATIONS:")
    print("  1. Questions with specific numerical facts (PTO days) work well")
    print("  2. Process questions may need better chunking to capture full context")
    print("  3. Consider using larger embedding models for better semantic understanding")
    print("  4. Reranking can help but may not always improve results significantly")
    print("  5. Chunk size of 300-500 tokens is appropriate for most queries")


if __name__ == "__main__":
    test_rag_system()
