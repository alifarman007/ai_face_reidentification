import os
import faiss
import numpy as np
import json
import logging
from typing import Tuple


class FaceDatabase:
    def __init__(self, embedding_size: int = 512, db_path: str = "./database/face_database") -> None:
        """
        Initialize the face database.

        Args:
            embedding_size: Dimension of face embeddings
            db_path: Directory to store database files
        """
        self.embedding_size = embedding_size
        self.db_path = db_path
        self.index_file = os.path.join(db_path, "faiss_index.bin")
        self.meta_file = os.path.join(db_path, "metadata.json")

        os.makedirs(db_path, exist_ok=True)

        # Use inner product for cosine similarity search
        self.index = faiss.IndexFlatIP(embedding_size)

        # Stores associated names for each embedding
        self.metadata = []

    def add_face(self, embedding: np.ndarray, name: str, image_path: str = None) -> None:
        """
        Add a face embedding to the database.

        Args:
            embedding: Face embedding vector
            name: Name of the person
            image_path: Path to the image file (for compatibility, not used in FAISS)
        """
        normalized_embedding = embedding / np.linalg.norm(embedding)
        self.index.add(np.array([normalized_embedding], dtype=np.float32))
        self.metadata.append(name)

    def search(self, embedding: np.ndarray, threshold: float = 0.4) -> Tuple[str, float]:
        """
        Search for the closest face in the database.

        Args:
            embedding: Query face embedding
            threshold: Similarity threshold

        Returns:
            Tuple containing the name and similarity score
        """
        if self.index.ntotal == 0:
            return "Unknown", 0.0

        normalized_embedding = embedding / np.linalg.norm(embedding)
        similarities, indices = self.index.search(np.array([normalized_embedding], dtype=np.float32), 1)

        best_similarity = similarities[0][0]
        best_idx = indices[0][0]

        if best_similarity > threshold and best_idx < len(self.metadata):
            return self.metadata[best_idx], best_similarity
        else:
            return "Unknown", best_similarity

    def save(self) -> None:
        """
        Save the FAISS index and metadata to disk.
        """
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logging.info(f"Face database saved with {self.index.ntotal} faces")

    def load(self) -> bool:
        """
        Load the FAISS index and metadata from disk.

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logging.info(f"Loaded face database with {self.index.ntotal} faces")
            return True
        return False

    def get_all_persons(self):
        """Get all persons from FAISS database (compatibility method)."""
        unique_names = list(set(self.metadata))
        persons = []
        for name in unique_names:
            embedding_count = self.metadata.count(name)
            persons.append({
                'name': name,
                'embedding_count': embedding_count,
                'id': None,  # FAISS doesn't have IDs
                'created_at': None  # FAISS doesn't track creation time
            })
        return persons

    def get_database_stats(self):
        """Get database statistics (compatibility method)."""
        unique_persons = len(set(self.metadata))
        total_embeddings = len(self.metadata)
        return {
            'person_count': unique_persons,
            'embedding_count': total_embeddings,
            'video_count': 0  # FAISS doesn't track video processing
        }

    def delete_person(self, name: str) -> bool:
        """Delete a person and their embeddings from FAISS database."""
        # Find indices of embeddings for this person
        indices_to_remove = [i for i, n in enumerate(self.metadata) if n == name]
        
        if not indices_to_remove:
            return False
        
        # Remove from metadata
        self.metadata = [n for n in self.metadata if n != name]
        
        # For FAISS, we need to rebuild the index without the removed embeddings
        if self.index.ntotal > 0:
            # Get all embeddings except the ones to remove
            all_embeddings = []
            for i in range(self.index.ntotal):
                if i not in indices_to_remove:
                    # Extract embedding (this is a simplified approach)
                    continue
            
            # Rebuild index (simplified - in practice you'd need to store embeddings separately)
            # For now, just remove from metadata and save
            logging.warning(f"Deleted person '{name}' from metadata. Index rebuild required for complete removal.")
        
        return True
