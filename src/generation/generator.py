"""
Main generator orchestrating retrieval + LLM synthesis.
"""

import re
import time
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.retriever import RAGRetriever
from src.generation.prompt_builder import PromptBuilder
from src.generation.llm_client import LLMClient


class RAGGenerator:
    """End-to-end RAG system: retrieve + generate."""
    
    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        llm_client: Optional[LLMClient] = None,
        top_k: int = 10
    ):
        """
        Initialize RAG generator.
        
        Args:
            retriever: RAGRetriever instance (or create default)
            llm_client: LLMClient instance (or create default)
            top_k: Number of chunks to retrieve
        """
        print("=" * 80)
        print("🤖 Initializing RAG Generator")
        print("=" * 80)
        
        # Initialize retriever
        if retriever is None:
            self.retriever = RAGRetriever()
        else:
            self.retriever = retriever
        
        # Initialize LLM client
        if llm_client is None:
            self.llm_client = LLMClient()
        else:
            self.llm_client = llm_client
        
        self.top_k = top_k
        
        print("=" * 80)
        print("✅ RAG Generator ready")
        print("=" * 80)
    
    def extract_citations(self, response: str) -> List[str]:
        """
        Extract review IDs from response citations.
        
        Args:
            response: LLM response text
        
        Returns:
            List of cited review IDs
        """
        # Pattern: [Review ID: xxx]
        pattern = r'\[Review ID:\s*([^\]]+)\]'
        matches = re.findall(pattern, response)
        return list(set(matches))  # Deduplicate
    
    def generate(
        self,
        query: str,
        enable_filtering: bool = True,
        return_contexts: bool = True
    ) -> Dict:
        """
        Generate answer for query.
        
        Pipeline:
        1. Retrieve relevant chunks
        2. Build prompt with context
        3. Call LLM
        4. Extract citations
        5. Return answer with metadata
        
        Args:
            query: Natural language query
            enable_filtering: Whether to apply metadata filters
            return_contexts: Whether to return retrieved contexts
        
        Returns:
            Dictionary with answer, contexts, citations, and metrics
        """
        start_time = time.time()
        
        # Step 1: Retrieve
        retrieval_start = time.time()
        retrieval_result = self.retriever.retrieve(
            query=query,
            top_k=self.top_k,
            enable_filtering=enable_filtering
        )
        retrieval_latency_ms = (time.time() - retrieval_start) * 1000
        
        results = retrieval_result['results']
        
        if not results:
            return {
                'query': query,
                'answer': "I couldn't find relevant reviews to answer your question.",
                'contexts': [],
                'citations': [],
                'filters_applied': retrieval_result.get('filters_applied', {}),
                'metrics': {
                    'retrieval_latency_ms': retrieval_latency_ms,
                    'generation_latency_ms': 0,
                    'total_latency_ms': retrieval_latency_ms,
                    'num_contexts': 0
                }
            }
        
        # Step 2: Build prompt
        system_prompt, user_prompt = PromptBuilder.build_prompt(query, results)
        
        # Step 3: Generate
        generation_start = time.time()
        llm_result = self.llm_client.generate(system_prompt, user_prompt)
        generation_latency_ms = (time.time() - generation_start) * 1000
        
        if not llm_result.get('response'):
            return {
                'query': query,
                'answer': f"Error generating response: {llm_result.get('error', 'Unknown error')}",
                'contexts': results if return_contexts else [],
                'citations': [],
                'filters_applied': retrieval_result.get('filters_applied', {}),
                'metrics': {
                    'retrieval_latency_ms': retrieval_latency_ms,
                    'generation_latency_ms': generation_latency_ms,
                    'total_latency_ms': (time.time() - start_time) * 1000,
                    'error': llm_result.get('error')
                }
            }
        
        # Step 4: Extract citations
        answer = llm_result['response']
        citations = self.extract_citations(answer)
        
        # Step 5: Prepare response
        total_latency_ms = (time.time() - start_time) * 1000
        
        return {
            'query': query,
            'answer': answer,
            'contexts': results if return_contexts else [],
            'citations': citations,
            'filters_applied': retrieval_result.get('filters_applied', {}),
            'metrics': {
                'retrieval_latency_ms': retrieval_latency_ms,
                'generation_latency_ms': generation_latency_ms,
                'total_latency_ms': total_latency_ms,
                'num_contexts': len(results),
                'num_citations': len(citations),
                'prompt_tokens': llm_result.get('prompt_tokens', 0),
                'completion_tokens': llm_result.get('completion_tokens', 0),
                'total_tokens': llm_result.get('total_tokens', 0)
            },
            'retrieval_pipeline': retrieval_result.get('pipeline', {})
        }


if __name__ == "__main__":
    # Test RAG generator
    print("\n" + "=" * 80)
    print("Testing RAG Generator")
    print("=" * 80 + "\n")
    
    # Initialize generator
    generator = RAGGenerator(top_k=5)
    
    # Test queries
    test_queries = [
        "What do visitors from Australia say about Disneyland in Hong Kong?",
        "Is spring a good time to visit Disneyland?",
        "Is the staff friendly?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)
        
        result = generator.generate(query, return_contexts=False)
        
        print(f"\n📊 Filters: {result['filters_applied']}")
        print(f"📈 Metrics:")
        print(f"   Retrieval: {result['metrics']['retrieval_latency_ms']:.0f}ms")
        print(f"   Generation: {result['metrics']['generation_latency_ms']:.0f}ms")
        print(f"   Total: {result['metrics']['total_latency_ms']:.0f}ms")
        print(f"   Contexts: {result['metrics']['num_contexts']}")
        print(f"   Citations: {result['metrics']['num_citations']}")
        print(f"   Tokens: {result['metrics']['total_tokens']}")
        
        print(f"\n💬 Answer:\n{result['answer']}")
        
        if result['citations']:
            print(f"\n📚 Citations: {', '.join(result['citations'])}")
        
        print("\n" + "=" * 80)

