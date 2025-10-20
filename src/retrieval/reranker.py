"""
Cross-encoder re-ranker for final result quality improvement.
Uses a cross-encoder model to re-score query-document pairs.
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict
import numpy as np


class CrossEncoderReranker:
    """Re-rank results using cross-encoder model."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize cross-encoder re-ranker.
        
        Args:
            model_name: Name of the cross-encoder model
        """
        print(f"Loading cross-encoder model: {model_name}...")
        try:
            self.model = CrossEncoder(model_name)
            self.enabled = True
            print(f"✅ Cross-encoder loaded")
        except Exception as e:
            print(f"⚠️  Cross-encoder loading failed: {e}")
            print(f"   Re-ranking will be skipped (using original scores)")
            self.enabled = False
    
    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Re-rank results using cross-encoder.
        
        Args:
            query: Query text
            results: List of result dictionaries with 'chunk_text'
            top_k: Number of top results to return
        
        Returns:
            Re-ranked list of results
        """
        if not self.enabled or not results:
            return results[:top_k]
        
        # Prepare query-document pairs
        pairs = [[query, result['chunk_text']] for result in results]
        
        # Get cross-encoder scores
        try:
            scores = self.model.predict(pairs)
            
            # Add scores to results
            for result, score in zip(results, scores):
                result['rerank_score'] = float(score)
            
            # Sort by re-rank score
            reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
            
            return reranked[:top_k]
            
        except Exception as e:
            print(f"⚠️  Re-ranking failed: {e}, returning original results")
            return results[:top_k]


if __name__ == "__main__":
    # Test re-ranker
    print("Testing Cross-Encoder Re-ranker...")
    
    reranker = CrossEncoderReranker()
    
    # Test data
    query = "Is the staff friendly?"
    
    results = [
        {'chunk_text': 'The staff at the park was very friendly and helpful.', 'score': 0.8},
        {'chunk_text': 'Great rides and attractions for all ages.', 'score': 0.7},
        {'chunk_text': 'Staff members were rude and unhelpful.', 'score': 0.6}
    ]
    
    reranked = reranker.rerank(query, results, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"\nRe-ranked results:")
    for i, result in enumerate(reranked):
        print(f"{i+1}. Rerank Score: {result.get('rerank_score', 'N/A'):.4f}")
        print(f"   Original Score: {result['score']:.4f}")
        print(f"   Text: {result['chunk_text']}")

