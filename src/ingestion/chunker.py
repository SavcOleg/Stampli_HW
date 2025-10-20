"""
Text chunking for long reviews.
Splits reviews into overlapping chunks to preserve context.
"""

from typing import List, Dict
import hashlib


class ReviewChunker:
    """Split long reviews into manageable, overlapping chunks."""
    
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100
    ):
        """
        Initialize chunker with size parameters.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            min_chunk_size: Minimum chunk size to keep (avoid tiny fragments)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def generate_chunk_id(self, review_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID from review ID and index."""
        combined = f"{review_id}_{chunk_index}"
        # Create a short hash for uniqueness
        hash_obj = hashlib.md5(combined.encode())
        return f"chunk_{hash_obj.hexdigest()[:12]}"
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The review text to chunk
        
        Returns:
            List of text chunks
        """
        if not text or len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is not the last chunk and we're not at text end,
            # try to break at a sentence or word boundary
            if end < len(text):
                # Look for period followed by space
                last_period = text[start:end].rfind('. ')
                if last_period > self.min_chunk_size:
                    end = start + last_period + 1
                else:
                    # Look for space (word boundary)
                    last_space = text[start:end].rfind(' ')
                    if last_space > self.min_chunk_size:
                        end = start + last_space
            
            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            # Move start forward, accounting for overlap
            start = end - self.chunk_overlap
            if start + self.min_chunk_size > len(text):
                break
        
        return chunks if chunks else [text]
    
    def chunk_review(
        self,
        review_data: Dict,
        review_id_field: str = 'review_id',
        text_field: str = 'review_text'
    ) -> List[Dict]:
        """
        Chunk a review and create chunk records with metadata.
        
        Args:
            review_data: Dictionary containing review data and metadata
            review_id_field: Name of the field containing review ID
            text_field: Name of the field containing review text
        
        Returns:
            List of chunk dictionaries, each containing chunk text and metadata
        """
        review_id = review_data.get(review_id_field)
        review_text = review_data.get(text_field, '')
        
        if not review_text:
            # Return single empty chunk if no text
            return [{
                **review_data,
                'chunk_id': self.generate_chunk_id(str(review_id), 0),
                'chunk_index': 0,
                'chunk_text': '',
                'total_chunks': 1
            }]
        
        text_chunks = self.chunk_text(review_text)
        total_chunks = len(text_chunks)
        
        chunk_records = []
        for idx, chunk_text in enumerate(text_chunks):
            chunk_record = {
                **review_data,  # Copy all metadata
                'chunk_id': self.generate_chunk_id(str(review_id), idx),
                'chunk_index': idx,
                'chunk_text': chunk_text,
                'total_chunks': total_chunks
            }
            chunk_records.append(chunk_record)
        
        return chunk_records
    
    def chunk_reviews_batch(
        self,
        reviews: List[Dict],
        review_id_field: str = 'review_id',
        text_field: str = 'review_text'
    ) -> List[Dict]:
        """
        Chunk multiple reviews in batch.
        
        Args:
            reviews: List of review dictionaries
            review_id_field: Name of the field containing review ID
            text_field: Name of the field containing review text
        
        Returns:
            Flattened list of all chunks from all reviews
        """
        all_chunks = []
        for review in reviews:
            chunks = self.chunk_review(review, review_id_field, text_field)
            all_chunks.extend(chunks)
        return all_chunks


if __name__ == "__main__":
    # Test the chunker
    chunker = ReviewChunker(chunk_size=100, chunk_overlap=20)
    
    test_review = {
        'review_id': 'test123',
        'park': 'Hong Kong',
        'country': 'Australia',
        'rating': 4,
        'review_text': (
            "This is a test review that is quite long and needs to be chunked. "
            "It contains multiple sentences and should be split properly. "
            "The chunker should preserve context by adding overlap between chunks. "
            "This helps maintain continuity when processing the text later."
        )
    }
    
    chunks = chunker.chunk_review(test_review)
    print(f"Generated {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"\nChunk {chunk['chunk_index']} (ID: {chunk['chunk_id']}):")
        print(f"  Text: {chunk['chunk_text'][:80]}...")
        print(f"  Total chunks: {chunk['total_chunks']}")

