"""
Hybrid search combining FAISS semantic search and BM25 keyword search.
Uses Reciprocal Rank Fusion (RRF) to merge results.
"""

import numpy as np
from typing import List, Tuple, Dict
import faiss
import pickle


class HybridSearcher:
    """Combine FAISS and BM25 results using Reciprocal Rank Fusion."""
    
    def __init__(
        self,
        faiss_index_path: str,
        bm25_index_path: str,
        metadata_path: str,
        embedder,
        faiss_weight: float = 0.6,
        bm25_weight: float = 0.4
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            faiss_index_path: Path to FAISS index
            bm25_index_path: Path to BM25 index
            metadata_path: Path to metadata pickle
            embedder: TextEmbedder instance for query embedding
            faiss_weight: Weight for FAISS results (0-1)
            bm25_weight: Weight for BM25 results (0-1)
        """
        self.embedder = embedder
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight
        
        # Load FAISS index
        print(f"Loading FAISS index from {faiss_index_path}...")
        self.faiss_index = faiss.read_index(faiss_index_path)
        
        # Load BM25 index
        print(f"Loading BM25 index from {bm25_index_path}...")
        with open(bm25_index_path, 'rb') as f:
            index_data = pickle.load(f)
            self.bm25_index = index_data['index']
            self.tokenized_corpus = index_data['tokenized_corpus']
        
        # Load metadata
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"✅ Loaded {len(self.metadata):,} chunks")
    
    def search_faiss(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        """
        Search using FAISS semantic similarity.
        
        Args:
            query: Query text
            k: Number of results to return
        
        Returns:
            List of (index, score) tuples
        """
        # Embed query
        query_embedding = self.embedder.encode_single(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Convert to list of tuples
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # Valid result
                # Convert L2 distance to similarity score (lower is better, so invert)
                score = 1.0 / (1.0 + dist)
                results.append((int(idx), float(score)))
        
        return results
    
    def search_bm25(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        """
        Search using BM25 keyword matching.
        
        Args:
            query: Query text
            k: Number of results to return
        
        Returns:
            List of (index, score) tuples
        """
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        # Return as list of tuples
        results = []
        for idx in top_k_indices:
            results.append((int(idx), float(scores[idx])))
        
        return results
    
    def reciprocal_rank_fusion(
        self,
        faiss_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Merge FAISS and BM25 results using Reciprocal Rank Fusion.
        
        RRF formula: score(d) = Σ 1 / (k + rank(d))
        
        Args:
            faiss_results: List of (index, score) from FAISS
            bm25_results: List of (index, score) from BM25
            k: Constant for RRF (typically 60)
        
        Returns:
            Merged and sorted list of (index, score) tuples
        """
        rrf_scores = {}
        
        # Add FAISS results
        for rank, (idx, score) in enumerate(faiss_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + \
                             self.faiss_weight / (k + rank + 1)
        
        # Add BM25 results
        for rank, (idx, score) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + \
                             self.bm25_weight / (k + rank + 1)
        
        # Sort by score descending
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results
    
    def apply_filters(
        self,
        results: List[Tuple[int, float]],
        filters: Dict
    ) -> List[Tuple[int, float]]:
        """
        Apply metadata filters to results.
        
        Args:
            results: List of (index, score) tuples
            filters: Dictionary of filters (park, country, season, etc.)
        
        Returns:
            Filtered list of (index, score) tuples
        """
        if not filters:
            return results
        
        filtered = []
        for idx, score in results:
            chunk = self.metadata.get(idx)
            if not chunk:
                continue
            
            # Check all filters
            matches = True
            for filter_key, filter_value in filters.items():
                chunk_value = chunk.get(filter_key)
                
                # Handle case-insensitive string comparison
                if isinstance(filter_value, str) and isinstance(chunk_value, str):
                    if filter_value.lower() != chunk_value.lower():
                        matches = False
                        break
                elif chunk_value != filter_value:
                    matches = False
                    break
            
            if matches:
                filtered.append((idx, score))
        
        return filtered
    
    def search(
        self,
        query: str,
        filters: Dict = None,
        k: int = 10,
        pre_filter_k: int = 50
    ) -> List[Dict]:
        """
        Perform hybrid search with optional filtering.
        
        Args:
            query: Query text
            filters: Optional metadata filters
            k: Number of final results
            pre_filter_k: Number of results to retrieve before filtering
        
        Returns:
            List of result dictionaries with chunk data and scores
        """
        # Search both indices
        faiss_results = self.search_faiss(query, k=pre_filter_k)
        bm25_results = self.search_bm25(query, k=pre_filter_k)
        
        # Merge with RRF
        merged_results = self.reciprocal_rank_fusion(
            faiss_results,
            bm25_results
        )
        
        # Apply filters if provided
        if filters:
            merged_results = self.apply_filters(merged_results, filters)
        
        # Get top-k and add metadata
        final_results = []
        for idx, score in merged_results[:k]:
            chunk = self.metadata.get(idx)
            if chunk:
                result = {
                    **chunk,
                    'index': idx,
                    'score': score
                }
                final_results.append(result)
        
        return final_results


if __name__ == "__main__":
    # Test hybrid search
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.retrieval.embedder import TextEmbedder
    
    print("Testing Hybrid Search...")
    
    embedder = TextEmbedder()
    
    searcher = HybridSearcher(
        faiss_index_path="data/indices/faiss_flat.index",
        bm25_index_path="data/indices/bm25.pkl",
        metadata_path="data/indices/metadata.pkl",
        embedder=embedder
    )
    
    # Test queries
    test_query = "Is the staff friendly?"
    print(f"\nQuery: {test_query}")
    
    results = searcher.search(test_query, k=5)
    
    print(f"\nTop 5 results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Score: {result['score']:.4f}")
        print(f"   Park: {result['park']}, Country: {result['country']}")
        print(f"   Text: {result['chunk_text'][:100]}...")

