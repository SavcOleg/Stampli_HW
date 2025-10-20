"""
FAISS index building and management.
Creates both Flat (exact) and HNSW (approximate) indices for vector search.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time


class FAISSIndexBuilder:
    """Build and manage FAISS indices for semantic search."""
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize FAISS index builder.
        
        Args:
            embedding_dim: Dimension of embeddings (384 for all-MiniLM-L6-v2)
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = {}
    
    def build_flat_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """
        Build a flat (exact search) index.
        Good for small datasets (<100K vectors) or as baseline.
        
        Args:
            embeddings: numpy array of shape (n_vectors, embedding_dim)
        
        Returns:
            FAISS Flat index
        """
        print(f"Building FAISS Flat index...")
        start_time = time.time()
        
        # Create index
        index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add vectors
        index.add(embeddings.astype('float32'))
        
        elapsed = time.time() - start_time
        print(f"✅ Built Flat index with {index.ntotal:,} vectors in {elapsed:.2f}s")
        
        return index
    
    def build_hnsw_index(
        self,
        embeddings: np.ndarray,
        M: int = 32,
        efConstruction: int = 200,
        efSearch: int = 64
    ) -> faiss.IndexHNSWFlat:
        """
        Build an HNSW (Hierarchical Navigable Small World) index.
        Provides fast approximate nearest neighbor search.
        
        Based on Context7 research:
        - M=32: good balance of accuracy and memory
        - efConstruction=200: build time vs quality tradeoff
        - efSearch=64: query time vs recall tradeoff
        
        Args:
            embeddings: numpy array of shape (n_vectors, embedding_dim)
            M: number of connections per element (higher = more accurate, more memory)
            efConstruction: size of dynamic candidate list during construction
            efSearch: size of dynamic candidate list during search
        
        Returns:
            FAISS HNSW index
        """
        print(f"Building FAISS HNSW index (M={M}, efC={efConstruction}, efS={efSearch})...")
        start_time = time.time()
        
        # Create HNSW index
        index = faiss.IndexHNSWFlat(self.embedding_dim, M)
        index.hnsw.efConstruction = efConstruction
        index.hnsw.efSearch = efSearch
        
        # Add vectors
        index.add(embeddings.astype('float32'))
        
        elapsed = time.time() - start_time
        print(f"✅ Built HNSW index with {index.ntotal:,} vectors in {elapsed:.2f}s")
        
        return index
    
    def save_index(
        self,
        index: faiss.Index,
        output_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save FAISS index to disk.
        
        Args:
            index: FAISS index to save
            output_path: Path to save index
            metadata: Optional metadata (chunk IDs, etc.)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(index, str(output_path))
        
        # Save metadata if provided
        if metadata:
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Saved index to {output_path} ({file_size_mb:.2f} MB)")
    
    def load_index(self, input_path: str) -> faiss.Index:
        """
        Load FAISS index from disk.
        
        Args:
            input_path: Path to load index from
        
        Returns:
            Loaded FAISS index
        """
        index = faiss.read_index(str(input_path))
        print(f"✅ Loaded index from {input_path} ({index.ntotal:,} vectors)")
        return index
    
    def load_metadata(self, metadata_path: str) -> Dict:
        """
        Load metadata from disk.
        
        Args:
            metadata_path: Path to metadata file
        
        Returns:
            Metadata dictionary
        """
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"✅ Loaded metadata ({len(metadata)} entries)")
        return metadata
    
    def search(
        self,
        index: faiss.Index,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search index for nearest neighbors.
        
        Args:
            index: FAISS index to search
            query_embedding: Query vector (1D or 2D array)
            k: Number of nearest neighbors to return
        
        Returns:
            Tuple of (distances, indices)
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = index.search(query_embedding.astype('float32'), k)
        
        return distances, indices
    
    def benchmark_search(
        self,
        index: faiss.Index,
        query_embeddings: np.ndarray,
        k: int = 10,
        n_queries: int = 100
    ):
        """
        Benchmark search performance.
        
        Args:
            index: FAISS index to benchmark
            query_embeddings: Sample queries for benchmarking
            k: Number of results per query
            n_queries: Number of queries to test
        """
        n_queries = min(n_queries, len(query_embeddings))
        sample_queries = query_embeddings[:n_queries]
        
        print(f"\n📊 Benchmarking search ({n_queries} queries, k={k})...")
        start_time = time.time()
        
        for query in sample_queries:
            self.search(index, query, k)
        
        elapsed = time.time() - start_time
        avg_latency_ms = (elapsed / n_queries) * 1000
        
        print(f"✅ Average search latency: {avg_latency_ms:.2f}ms per query")
        print(f"   Throughput: {n_queries / elapsed:.1f} queries/sec")


if __name__ == "__main__":
    # Test FAISS index building
    print("Testing FAISS index builder...")
    
    # Create dummy embeddings
    n_vectors = 1000
    embedding_dim = 384
    embeddings = np.random.rand(n_vectors, embedding_dim).astype('float32')
    
    builder = FAISSIndexBuilder(embedding_dim=embedding_dim)
    
    # Build Flat index
    flat_index = builder.build_flat_index(embeddings)
    
    # Build HNSW index
    hnsw_index = builder.build_hnsw_index(embeddings)
    
    # Test search
    query = np.random.rand(embedding_dim).astype('float32')
    distances, indices = builder.search(hnsw_index, query, k=5)
    
    print(f"\nTest search results:")
    print(f"  Top 5 indices: {indices[0]}")
    print(f"  Top 5 distances: {distances[0]}")
    
    # Benchmark
    builder.benchmark_search(hnsw_index, embeddings[:100], k=10, n_queries=100)

