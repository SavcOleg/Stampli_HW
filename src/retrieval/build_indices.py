"""
Orchestrate index building: embeddings, FAISS, and BM25.
Main entry point for creating all retrieval indices.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

from src.retrieval.embedder import TextEmbedder
from src.retrieval.faiss_index import FAISSIndexBuilder
from src.retrieval.bm25_index import BM25IndexBuilder


class IndexBuildPipeline:
    """Build all retrieval indices from processed chunks."""
    
    def __init__(
        self,
        chunks_path: str = "data/processed/chunks.parquet",
        embeddings_path: str = "data/processed/embeddings.npy",
        faiss_flat_path: str = "data/indices/faiss_flat.index",
        faiss_hnsw_path: str = "data/indices/faiss_hnsw.index",
        bm25_path: str = "data/indices/bm25.pkl",
        metadata_path: str = "data/indices/metadata.pkl",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize index build pipeline.
        
        Args:
            chunks_path: Path to chunks parquet file
            embeddings_path: Path to save embeddings
            faiss_flat_path: Path to save FAISS Flat index
            faiss_hnsw_path: Path to save FAISS HNSW index
            bm25_path: Path to save BM25 index
            metadata_path: Path to save chunk metadata
            embedding_model: Name of embedding model
        """
        self.chunks_path = Path(chunks_path)
        self.embeddings_path = Path(embeddings_path)
        self.faiss_flat_path = Path(faiss_flat_path)
        self.faiss_hnsw_path = Path(faiss_hnsw_path)
        self.bm25_path = Path(bm25_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_model = embedding_model
        
        # Ensure directories exist
        for path in [self.embeddings_path, self.faiss_flat_path, 
                     self.faiss_hnsw_path, self.bm25_path, self.metadata_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_chunks(self) -> pd.DataFrame:
        """Load processed chunks from parquet."""
        print(f"📖 Loading chunks from {self.chunks_path}...")
        
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {self.chunks_path}\n"
                "Run 'make ingest' first to process the data."
            )
        
        df = pd.read_parquet(self.chunks_path)
        print(f"✅ Loaded {len(df):,} chunks")
        
        return df
    
    def generate_embeddings(self, chunks_df: pd.DataFrame) -> np.ndarray:
        """
        Generate embeddings for all chunks.
        
        Args:
            chunks_df: DataFrame with chunks
        
        Returns:
            numpy array of embeddings
        """
        print(f"\n🔄 Generating embeddings...")
        
        # Initialize embedder
        embedder = TextEmbedder(model_name=self.embedding_model)
        
        # Extract chunk texts
        chunk_texts = chunks_df['chunk_text'].tolist()
        
        # Generate embeddings
        embeddings = embedder.encode_batch(chunk_texts, show_progress=True)
        
        # Save embeddings
        metadata = {
            'model_name': self.embedding_model,
            'embedding_dim': embedder.embedding_dim,
            'n_chunks': len(chunk_texts)
        }
        embedder.save_embeddings(embeddings, self.embeddings_path, metadata)
        
        return embeddings
    
    def build_faiss_indices(self, embeddings: np.ndarray):
        """
        Build FAISS Flat and HNSW indices.
        
        Args:
            embeddings: numpy array of embeddings
        """
        print(f"\n🔄 Building FAISS indices...")
        
        embedding_dim = embeddings.shape[1]
        builder = FAISSIndexBuilder(embedding_dim=embedding_dim)
        
        # Build Flat index if not exists
        if not self.faiss_flat_path.exists():
            flat_index = builder.build_flat_index(embeddings)
            builder.save_index(flat_index, self.faiss_flat_path)
        else:
            print(f"✅ FAISS Flat index already exists, skipping...")
        
        # Build HNSW index with error handling (may cause segfault on some systems)
        try:
            if not self.faiss_hnsw_path.exists():
                print(f"Building HNSW index (this may take a while)...")
                hnsw_index = builder.build_hnsw_index(
                    embeddings,
                    M=16,  # Reduced from 32 to use less memory
                    efConstruction=100,  # Reduced from 200
                    efSearch=32  # Reduced from 64
                )
                builder.save_index(hnsw_index, self.faiss_hnsw_path)
                
                # Benchmark HNSW
                builder.benchmark_search(hnsw_index, embeddings[:100], k=10, n_queries=100)
            else:
                print(f"✅ FAISS HNSW index already exists, skipping...")
        except Exception as e:
            print(f"⚠️  HNSW index build failed (will use Flat index): {str(e)}")
            print(f"   This is OK - Flat index provides exact search results.")
    
    def build_bm25_index(self, chunks_df: pd.DataFrame):
        """
        Build BM25 index for keyword search.
        
        Args:
            chunks_df: DataFrame with chunks
        """
        print(f"\n🔄 Building BM25 index...")
        
        builder = BM25IndexBuilder()
        
        # Extract chunk texts
        chunk_texts = chunks_df['chunk_text'].tolist()
        
        # Build index
        builder.build_index(chunk_texts)
        
        # Save index
        builder.save_index(self.bm25_path)
        
        # Benchmark
        test_queries = [
            "staff friendly",
            "food expensive",
            "rides crowded",
            "spring weather",
            "Hong Kong Disneyland"
        ]
        builder.benchmark_search(test_queries, k=10)
    
    def save_metadata(self, chunks_df: pd.DataFrame):
        """
        Save chunk metadata for retrieval.
        
        Args:
            chunks_df: DataFrame with chunks
        """
        print(f"\n💾 Saving chunk metadata...")
        
        # Create metadata: index → chunk data
        metadata = {}
        for idx, row in chunks_df.iterrows():
            metadata[idx] = {
                'chunk_id': row['chunk_id'],
                'review_id': row['review_id'],
                'chunk_text': row['chunk_text'],
                'park': row['park'],
                'country': row['country'],
                'season': row['season'],
                'month_name': row['month_name'],
                'rating': row['rating'],
                'rating_tier': row['rating_tier'],
                'topics': row['topics'].tolist() if isinstance(row['topics'], np.ndarray) else row['topics'],
                'year': row['year'],
                'month': row['month']
            }
        
        import pickle
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        file_size_mb = self.metadata_path.stat().st_size / (1024 * 1024)
        print(f"✅ Saved metadata for {len(metadata):,} chunks ({file_size_mb:.2f} MB)")
    
    def print_summary(self, chunks_df: pd.DataFrame):
        """Print summary of built indices."""
        print(f"\n" + "=" * 80)
        print("📊 Index Build Summary")
        print("=" * 80)
        
        # File sizes
        files = {
            'Embeddings': self.embeddings_path,
            'FAISS Flat': self.faiss_flat_path,
            'FAISS HNSW': self.faiss_hnsw_path,
            'BM25': self.bm25_path,
            'Metadata': self.metadata_path
        }
        
        print("\nGenerated Files:")
        total_size_mb = 0
        for name, path in files.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                total_size_mb += size_mb
                print(f"  {name:15} {path} ({size_mb:.2f} MB)")
        
        print(f"\n  Total Size: {total_size_mb:.2f} MB")
        print(f"  Total Chunks Indexed: {len(chunks_df):,}")
        print(f"  Embedding Dimension: 384")
    
    def run(self, skip_embeddings: bool = False):
        """
        Run the full index building pipeline.
        
        Args:
            skip_embeddings: If True, load existing embeddings instead of regenerating
        """
        print("=" * 80)
        print("🚀 Starting Index Building Pipeline")
        print("=" * 80)
        
        try:
            # Load chunks
            chunks_df = self.load_chunks()
            
            # Generate or load embeddings
            if skip_embeddings and self.embeddings_path.exists():
                print(f"\n📖 Loading existing embeddings from {self.embeddings_path}...")
                embeddings = np.load(self.embeddings_path)
                print(f"✅ Loaded embeddings (shape: {embeddings.shape})")
            else:
                embeddings = self.generate_embeddings(chunks_df)
            
            # Build FAISS indices
            self.build_faiss_indices(embeddings)
            
            # Build BM25 index
            self.build_bm25_index(chunks_df)
            
            # Save metadata
            self.save_metadata(chunks_df)
            
            # Print summary
            self.print_summary(chunks_df)
            
            print("\n" + "=" * 80)
            print("✅ Index building completed successfully!")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n❌ Error during index building: {str(e)}", file=sys.stderr)
            raise


def main():
    """Main entry point for index building."""
    pipeline = IndexBuildPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()

