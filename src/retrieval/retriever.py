"""
Main retriever orchestrating the full pipeline:
Query Parser → Hybrid Search → MMR → Re-ranker
"""

from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.embedder import TextEmbedder
from src.retrieval.query_parser import QueryParser
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.mmr import MMRDiversifier
from src.retrieval.reranker import CrossEncoderReranker


class RAGRetriever:
    """End-to-end retrieval pipeline for RAG system."""
    
    def __init__(
        self,
        faiss_index_path: str = "data/indices/faiss_flat.index",
        bm25_index_path: str = "data/indices/bm25.pkl",
        metadata_path: str = "data/indices/metadata.pkl",
        embeddings_path: str = "data/processed/embeddings.npy",
        lookup_dir: str = "data/lookup",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        mmr_lambda: float = 0.6,
        enable_mmr: bool = True,
        enable_reranker: bool = True
    ):
        """
        Initialize full retrieval pipeline.
        
        Args:
            faiss_index_path: Path to FAISS index
            bm25_index_path: Path to BM25 index
            metadata_path: Path to metadata
            embeddings_path: Path to embeddings (for MMR)
            lookup_dir: Directory with lookup YAMLs
            embedding_model: Embedding model name
            reranker_model: Re-ranker model name
            mmr_lambda: MMR diversity parameter
            enable_mmr: Whether to use MMR diversification
            enable_reranker: Whether to use cross-encoder re-ranking
        """
        print("=" * 80)
        print("🚀 Initializing RAG Retriever")
        print("=" * 80)
        
        # Initialize components
        self.embedder = TextEmbedder(model_name=embedding_model)
        self.query_parser = QueryParser(lookup_dir=lookup_dir)
        self.hybrid_searcher = HybridSearcher(
            faiss_index_path=faiss_index_path,
            bm25_index_path=bm25_index_path,
            metadata_path=metadata_path,
            embedder=self.embedder
        )
        
        # Load embeddings for MMR
        self.enable_mmr = enable_mmr
        if self.enable_mmr:
            print(f"Loading embeddings for MMR from {embeddings_path}...")
            self.embeddings = np.load(embeddings_path)
            self.mmr = MMRDiversifier(lambda_param=mmr_lambda)
            print(f"✅ MMR enabled (λ={mmr_lambda})")
        else:
            self.embeddings = None
            self.mmr = None
            print("⚠️  MMR disabled")
        
        # Initialize re-ranker
        self.enable_reranker = enable_reranker
        if self.enable_reranker:
            self.reranker = CrossEncoderReranker(model_name=reranker_model)
        else:
            self.reranker = None
            print("⚠️  Re-ranker disabled")
        
        print("=" * 80)
        print("✅ RAG Retriever initialized successfully")
        print("=" * 80)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        pre_mmr_k: int = 50,
        pre_rerank_k: int = 20,
        enable_filtering: bool = True
    ) -> Dict:
        """
        Full retrieval pipeline.
        
        Pipeline:
        1. Parse query for intents/filters
        2. Hybrid search (FAISS + BM25)
        3. Apply metadata filters
        4. MMR diversification (optional)
        5. Cross-encoder re-ranking (optional)
        
        Args:
            query: Natural language query
            top_k: Final number of results to return
            pre_mmr_k: Number of results before MMR
            pre_rerank_k: Number of results before re-ranking
            enable_filtering: Whether to apply metadata filters
        
        Returns:
            Dictionary with results and metadata
        """
        # Step 1: Parse query
        parsed = self.query_parser.parse(query)
        filters = parsed['filters'] if enable_filtering else {}
        
        # Step 2: Hybrid search
        hybrid_results = self.hybrid_searcher.search(
            query=query,
            filters=filters,
            k=pre_mmr_k,
            pre_filter_k=100  # Get more before filtering
        )
        
        if not hybrid_results:
            return {
                'query': query,
                'parsed_query': parsed,
                'results': [],
                'total_results': 0
            }
        
        # Step 3: MMR diversification (if enabled)
        if self.enable_mmr and self.mmr and len(hybrid_results) > pre_rerank_k:
            # Get embeddings for candidates
            candidate_indices = [r['index'] for r in hybrid_results]
            candidate_embeddings = [self.embeddings[idx] for idx in candidate_indices]
            
            # Get query embedding
            query_embedding = self.embedder.encode_single(query)
            
            # Apply MMR
            mmr_results = self.mmr.diversify(
                query_embedding=query_embedding,
                candidate_embeddings=candidate_embeddings,
                candidate_results=hybrid_results,
                k=pre_rerank_k
            )
        else:
            mmr_results = hybrid_results[:pre_rerank_k]
        
        # Step 4: Cross-encoder re-ranking (if enabled)
        if self.enable_reranker and self.reranker and self.reranker.enabled:
            final_results = self.reranker.rerank(
                query=query,
                results=mmr_results,
                top_k=top_k
            )
        else:
            final_results = mmr_results[:top_k]
        
        # Return results with metadata
        return {
            'query': query,
            'parsed_query': parsed,
            'results': final_results,
            'total_results': len(final_results),
            'filters_applied': filters,
            'pipeline': {
                'hybrid_results': len(hybrid_results),
                'after_mmr': len(mmr_results),
                'final': len(final_results)
            }
        }


if __name__ == "__main__":
    # Test the full retriever
    print("\n" + "=" * 80)
    print("Testing Full RAG Retriever")
    print("=" * 80 + "\n")
    
    # Initialize retriever
    retriever = RAGRetriever()
    
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
        
        result = retriever.retrieve(query, top_k=3)
        
        print(f"\nFilters applied: {result['filters_applied']}")
        print(f"Pipeline: {result['pipeline']}")
        print(f"\nTop {len(result['results'])} results:")
        
        for i, res in enumerate(result['results']):
            print(f"\n{i+1}. Park: {res['park']}, Country: {res['country']}, Season: {res['season']}")
            print(f"   Score: {res.get('score', 0):.4f}, Rerank: {res.get('rerank_score', 'N/A')}")
            print(f"   Text: {res['chunk_text'][:150]}...")

