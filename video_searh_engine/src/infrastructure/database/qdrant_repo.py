# Concrete implementation for data persistence using Qdrant
import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

#modelo e interfaz
from src.domain.interfaces.repository import VectorRepositoryInterface
from src.domain.models import SearchResult

class QdrantVectorRepository(VectorRepositoryInterface):
    """implementacion qdrant en local"""

    def __init__(self, storage_path:str = "./qdrant_data",collection_name:str = "video_frames",vector_size:int = 768):
        #inicializar la base de datos local
        self.client = QdrantClient(path= storage_path)
        self.collection = collection_name
        self.ensure_collection = vector_size 


    def _ensure_collection(self, vector_size: int) -> None:
        """Crea la colección local si aún no existe."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

        return None
    def save_vectors(self,video_name: str,frame_ids: List[str],timestamps: List[float],vectors: List[float]):

        return None

    def search_similar(self,query_vector: List[float],limit: int = 5) -> List[SearchResult]:

        return None
        


if __name__ == "__main__":
    # Ejemplo de uso
    repo = QdrantVectorRepository(storage_path="./qdrant_data", collection_name="video_frames", vector_size=768)
    repo._ensure_collection(vector_size=768)

