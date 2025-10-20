"""
Text embedding using sentence-transformers.
Converts text chunks into dense vector representations for semantic search.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
import pickle


class TextEmbedder:
    """Generate embeddings for text chunks using sentence-transformers."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize embedder with model.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode a batch of texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings to unit length
        
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single text into embedding.
        
        Args:
            text: Text string to encode
            normalize: Whether to normalize embedding
        
        Returns:
            numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        return embedding
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        output_path: str,
        metadata: Optional[dict] = None
    ):
        """
        Save embeddings to disk.
        
        Args:
            embeddings: numpy array of embeddings
            output_path: Path to save embeddings
            metadata: Optional metadata to save alongside
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings as numpy array
        np.save(output_path, embeddings)
        
        # Save metadata if provided
        if metadata:
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Saved embeddings to {output_path} ({file_size_mb:.2f} MB)")
    
    def load_embeddings(self, input_path: str) -> np.ndarray:
        """
        Load embeddings from disk.
        
        Args:
            input_path: Path to load embeddings from
        
        Returns:
            numpy array of embeddings
        """
        embeddings = np.load(input_path)
        print(f"✅ Loaded embeddings from {input_path} (shape: {embeddings.shape})")
        return embeddings


if __name__ == "__main__":
    # Test the embedder
    embedder = TextEmbedder()
    
    test_texts = [
        "The staff at Disneyland was very friendly.",
        "Food prices are quite expensive in the park.",
        "The rides were amazing and worth the wait."
    ]
    
    print("\nTesting embedding generation...")
    embeddings = embedder.encode_batch(test_texts, show_progress=False)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first 5 dims): {embeddings[0][:5]}")
    
    # Test single encoding
    single_embedding = embedder.encode_single("Test query")
    print(f"Single embedding shape: {single_embedding.shape}")

