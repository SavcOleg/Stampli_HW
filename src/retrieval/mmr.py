"""
Maximal Marginal Relevance (MMR) for result diversification.
Reduces redundancy by selecting results that are relevant but diverse.
"""

import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity


class MMRDiversifier:
    """Apply MMR to diversify search results."""
    
    def __init__(self, lambda_param: float = 0.6):
        """
        Initialize MMR with lambda parameter.
        
        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
                         Typical value: 0.5-0.7
        """
        self.lambda_param = lambda_param
    
    def diversify(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        candidate_results: List[Dict],
        k: int = 20
    ) -> List[Dict]:
        """
        Apply MMR to select diverse results.
        
        MMR formula:
        MMR = argmax[Di in R\S] [λ * Sim(Di, Q) - (1-λ) * max[Dj in S] Sim(Di, Dj)]
        
        Where:
        - Di: candidate document
        - Q: query
        - S: selected documents
        - R: candidate documents
        - λ: relevance vs diversity trade-off
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            candidate_results: List of candidate result dicts
            k: Number of results to select
        
        Returns:
            List of k diversified results
        """
        if not candidate_results or k == 0:
            return []
        
        if len(candidate_results) <= k:
            return candidate_results
        
        # Convert to numpy arrays
        query_emb = query_embedding.reshape(1, -1)
        candidate_embs = np.array(candidate_embeddings)
        
        # Compute relevance scores (similarity to query)
        relevance_scores = cosine_similarity(candidate_embs, query_emb).flatten()
        
        # Initialize
        selected_indices = []
        remaining_indices = list(range(len(candidate_results)))
        
        # Select first result (highest relevance)
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select remaining k-1 results
        while len(selected_indices) < k and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance: similarity to query
                relevance = relevance_scores[idx]
                
                # Diversity: max similarity to already selected documents
                selected_embs = candidate_embs[selected_indices]
                candidate_emb = candidate_embs[idx].reshape(1, -1)
                similarities = cosine_similarity(candidate_emb, selected_embs).flatten()
                max_similarity = np.max(similarities)
                
                # MMR score
                mmr_score = (self.lambda_param * relevance - 
                           (1 - self.lambda_param) * max_similarity)
                mmr_scores.append(mmr_score)
            
            # Select candidate with highest MMR score
            best_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Return selected results
        return [candidate_results[i] for i in selected_indices]


if __name__ == "__main__":
    # Test MMR
    mmr = MMRDiversifier(lambda_param=0.6)
    
    # Create dummy data
    query_emb = np.random.rand(384)
    
    # Create some similar and diverse embeddings
    candidate_embs = []
    for i in range(10):
        if i < 5:
            # Similar to query
            emb = query_emb + np.random.rand(384) * 0.1
        else:
            # Different from query
            emb = np.random.rand(384)
        candidate_embs.append(emb)
    
    candidate_results = [{'id': i, 'text': f'Result {i}'} for i in range(10)]
    
    # Apply MMR
    diversified = mmr.diversify(query_emb, candidate_embs, candidate_results, k=5)
    
    print("MMR Diversification Test:")
    print(f"Selected {len(diversified)} results from {len(candidate_results)} candidates")
    print(f"Selected IDs: {[r['id'] for r in diversified]}")

