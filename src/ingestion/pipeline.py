"""
Data ingestion pipeline for Disney reviews.
Orchestrates feature extraction, chunking, and storage.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import sys
from tqdm import tqdm

from src.ingestion.feature_extractor import FeatureExtractor
from src.ingestion.chunker import ReviewChunker


class IngestionPipeline:
    """End-to-end data ingestion pipeline."""
    
    def __init__(
        self,
        input_csv: str = "data/raw/DisneylandReviews.csv",
        output_parquet: str = "data/processed/chunks.parquet",
        lookup_dir: str = "data/lookup",
        chunk_size: int = 400,
        chunk_overlap: int = 100
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            input_csv: Path to input CSV file
            output_parquet: Path to output Parquet file
            lookup_dir: Directory containing lookup YAML files
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.input_csv = Path(input_csv)
        self.output_parquet = Path(output_parquet)
        self.lookup_dir = lookup_dir
        
        self.feature_extractor = FeatureExtractor(lookup_dir=lookup_dir)
        self.chunker = ReviewChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Ensure output directory exists
        self.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    
    def load_reviews(self) -> pd.DataFrame:
        """Load reviews from CSV file."""
        print(f"📖 Loading reviews from {self.input_csv}...")
        
        if not self.input_csv.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_csv}")
        
        # Try different encodings to handle various text formats
        try:
            df = pd.read_csv(self.input_csv, encoding='utf-8')
        except UnicodeDecodeError:
            print("   UTF-8 failed, trying latin-1 encoding...")
            df = pd.read_csv(self.input_csv, encoding='latin-1')
        
        print(f"✅ Loaded {len(df):,} reviews")
        
        return df
    
    def process_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process reviews: extract features and chunk.
        
        Args:
            df: DataFrame with raw review data
        
        Returns:
            DataFrame with processed chunks
        """
        print(f"\n🔄 Processing reviews...")
        
        all_chunks = []
        
        # Process each review with progress bar
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            # Extract features
            features = self.feature_extractor.extract_all_features(row.to_dict())
            
            # Chunk review
            chunks = self.chunker.chunk_review(features)
            all_chunks.extend(chunks)
        
        # Convert to DataFrame
        chunks_df = pd.DataFrame(all_chunks)
        
        print(f"✅ Generated {len(chunks_df):,} chunks from {len(df):,} reviews")
        print(f"   Average chunks per review: {len(chunks_df)/len(df):.2f}")
        
        return chunks_df
    
    def save_chunks(self, chunks_df: pd.DataFrame):
        """Save chunks to Parquet file with compression."""
        print(f"\n💾 Saving chunks to {self.output_parquet}...")
        
        # Save with compression
        chunks_df.to_parquet(
            self.output_parquet,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        # Get file size
        file_size_mb = self.output_parquet.stat().st_size / (1024 * 1024)
        print(f"✅ Saved {len(chunks_df):,} chunks ({file_size_mb:.2f} MB)")
    
    def print_sample_chunks(self, chunks_df: pd.DataFrame, n: int = 5):
        """Print sample chunks for validation."""
        print(f"\n📊 Sample of {n} chunks:")
        print("-" * 80)
        
        for idx, row in chunks_df.head(n).iterrows():
            print(f"\nChunk {idx + 1}:")
            print(f"  Chunk ID: {row['chunk_id']}")
            print(f"  Review ID: {row['review_id']}")
            print(f"  Park: {row['park']}")
            print(f"  Country: {row['country']}")
            print(f"  Season: {row['season']} ({row['month_name']})")
            print(f"  Rating: {row['rating']} ({row['rating_tier']})")
            print(f"  Topics: {', '.join(row['topics']) if row['topics'] else 'None'}")
            print(f"  Chunk: {row['chunk_index'] + 1}/{row['total_chunks']}")
            print(f"  Text: {row['chunk_text'][:100]}...")
        
        print("-" * 80)
    
    def print_statistics(self, chunks_df: pd.DataFrame):
        """Print dataset statistics."""
        print(f"\n📈 Dataset Statistics:")
        print(f"  Total chunks: {len(chunks_df):,}")
        print(f"  Unique reviews: {chunks_df['review_id'].nunique():,}")
        print(f"  Average chunk length: {chunks_df['chunk_text'].str.len().mean():.0f} chars")
        print(f"\n  Parks:")
        for park, count in chunks_df['park'].value_counts().items():
            print(f"    {park}: {count:,} chunks")
        print(f"\n  Rating distribution:")
        for rating in sorted(chunks_df['rating'].unique()):
            count = (chunks_df['rating'] == rating).sum()
            print(f"    {rating} stars: {count:,} chunks")
        print(f"\n  Top 5 countries:")
        for country, count in chunks_df['country'].value_counts().head(5).items():
            print(f"    {country}: {count:,} chunks")
        print(f"\n  Seasons:")
        for season, count in chunks_df['season'].value_counts().items():
            print(f"    {season}: {count:,} chunks")
    
    def run(self, show_samples: bool = True, show_stats: bool = True):
        """
        Run the full ingestion pipeline.
        
        Args:
            show_samples: Whether to print sample chunks
            show_stats: Whether to print statistics
        """
        print("=" * 80)
        print("🚀 Starting Disney Reviews Ingestion Pipeline")
        print("=" * 80)
        
        try:
            # Load reviews
            df = self.load_reviews()
            
            # Process reviews (extract features and chunk)
            chunks_df = self.process_reviews(df)
            
            # Save to Parquet
            self.save_chunks(chunks_df)
            
            # Show samples and statistics
            if show_samples:
                self.print_sample_chunks(chunks_df)
            
            if show_stats:
                self.print_statistics(chunks_df)
            
            print("\n" + "=" * 80)
            print("✅ Ingestion pipeline completed successfully!")
            print("=" * 80)
            
            return chunks_df
            
        except Exception as e:
            print(f"\n❌ Error during ingestion: {str(e)}", file=sys.stderr)
            raise


def main():
    """Main entry point for ingestion pipeline."""
    pipeline = IngestionPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()

