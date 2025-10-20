"""
BM25 index for lexical/keyword-based search.
Complements semantic search with exact term matching.
"""

from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import time


class BM25IndexBuilder:
    """Build and manage BM25 index for keyword search."""
    
    def __init__(self):
        """Initialize BM25 index builder."""
        self.index = None
        self.tokenized_corpus = []
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase and split by whitespace.
        
        Args:
            text: Input text to tokenize
        
        Returns:
            List of tokens
        """
        return text.lower().split()
    
    def build_index(self, texts: List[str]) -> BM25Okapi:
        """
        Build BM25 index from corpus of texts.
        
        Args:
            texts: List of text documents
        
        Returns:
            BM25 index
        """
        print(f"Building BM25 index from {len(texts):,} documents...")
        start_time = time.time()
        
        # Tokenize corpus
        self.tokenized_corpus = [self.tokenize(text) for text in texts]
        
        # Build index
        self.index = BM25Okapi(self.tokenized_corpus)
        
        elapsed = time.time() - start_time
        print(f"✅ Built BM25 index in {elapsed:.2f}s")
        
        return self.index
    
    def search(self, query: str, k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Search BM25 index for top-k matching documents.
        
        Args:
            query: Query text
            k: Number of results to return
        
        Returns:
            Tuple of (indices, scores)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Tokenize query
        tokenized_query = self.tokenize(query)
        
        # Get scores for all documents
        scores = self.index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = scores.argsort()[-k:][::-1]
        top_k_scores = scores[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()
    
    def save_index(self, output_path: str):
        """
        Save BM25 index to disk.
        
        Args:
            output_path: Path to save index
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'index': self.index,
            'tokenized_corpus': self.tokenized_corpus
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Saved BM25 index to {output_path} ({file_size_mb:.2f} MB)")
    
    def load_index(self, input_path: str):
        """
        Load BM25 index from disk.
        
        Args:
            input_path: Path to load index from
        """
        with open(input_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.index = index_data['index']
        self.tokenized_corpus = index_data['tokenized_corpus']
        
        print(f"✅ Loaded BM25 index from {input_path} ({len(self.tokenized_corpus):,} docs)")
    
    def benchmark_search(self, queries: List[str], k: int = 10):
        """
        Benchmark BM25 search performance.
        
        Args:
            queries: List of test queries
            k: Number of results per query
        """
        print(f"\n📊 Benchmarking BM25 search ({len(queries)} queries, k={k})...")
        start_time = time.time()
        
        for query in queries:
            self.search(query, k)
        
        elapsed = time.time() - start_time
        avg_latency_ms = (elapsed / len(queries)) * 1000
        
        print(f"✅ Average search latency: {avg_latency_ms:.2f}ms per query")
        print(f"   Throughput: {len(queries) / elapsed:.1f} queries/sec")


if __name__ == "__main__":
    # Test BM25 index
    print("Testing BM25 index builder...")
    
    # Sample corpus
    corpus = [
        "The staff at Disneyland was very friendly and helpful",
        "Food prices are quite expensive in the park",
        "The rides were amazing and worth the long wait times",
        "Hong Kong Disneyland is smaller than California",
        "Spring is a great time to visit with pleasant weather"
    ]
    
    builder = BM25IndexBuilder()
    builder.build_index(corpus)
    
    # Test search
    query = "staff friendly helpful"
    indices, scores = builder.search(query, k=3)
    
    print(f"\nQuery: '{query}'")
    print("Top 3 results:")
    for idx, (doc_idx, score) in enumerate(zip(indices, scores)):
        print(f"  {idx + 1}. Score: {score:.3f} - {corpus[doc_idx]}")
    
    # Benchmark
    test_queries = ["staff", "food expensive", "rides wait", "Hong Kong", "spring weather"]
    builder.benchmark_search(test_queries, k=3)

