"""
Safe index builder - builds only BM25 and metadata (skips FAISS if already done).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.bm25_index import BM25IndexBuilder


def build_bm25_and_metadata():
    """Build BM25 index and save metadata."""
    
    print("=" * 80)
    print("🚀 Building BM25 Index and Metadata")
    print("=" * 80)
    
    # Paths
    chunks_path = Path("data/processed/chunks.parquet")
    bm25_path = Path("data/indices/bm25.pkl")
    metadata_path = Path("data/indices/metadata.pkl")
    
    # Load chunks
    print(f"\n📖 Loading chunks from {chunks_path}...")
    chunks_df = pd.read_parquet(chunks_path)
    print(f"✅ Loaded {len(chunks_df):,} chunks")
    
    # Build BM25 index
    print(f"\n🔄 Building BM25 index...")
    bm25_builder = BM25IndexBuilder()
    chunk_texts = chunks_df['chunk_text'].tolist()
    bm25_builder.build_index(chunk_texts)
    bm25_builder.save_index(bm25_path)
    
    # Benchmark
    test_queries = [
        "staff friendly",
        "food expensive",
        "rides crowded",
        "spring weather",
        "Hong Kong Disneyland"
    ]
    bm25_builder.benchmark_search(test_queries, k=10)
    
    # Save metadata
    print(f"\n💾 Saving chunk metadata...")
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
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    file_size_mb = metadata_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved metadata for {len(metadata):,} chunks ({file_size_mb:.2f} MB)")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("📊 Build Summary")
    print("=" * 80)
    
    files = {
        'BM25 Index': bm25_path,
        'Metadata': metadata_path
    }
    
    total_size_mb = 0
    for name, path in files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            print(f"  {name:15} {size_mb:.2f} MB")
    
    print(f"\n  Total: {total_size_mb:.2f} MB")
    print(f"\n✅ BM25 and metadata build completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    build_bm25_and_metadata()

