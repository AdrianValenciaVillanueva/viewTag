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
        self._ensure_collection(vector_size)

    # Asegurar que la colección exista
    def _ensure_collection(self, vector_size: int) -> None:
        """Crea la colección local si aún no existe."""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

    # Guardar vectores en la base de datos
    def save_vectors(self,video_name: str,frame_ids: List[str],timestamps: List[float],vectors: List[List[float]]) -> None:

        points = []

        for f_id, ts, vec, in zip(frame_ids, timestamps, vectors):
            # Genera un ID único para cada punto
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_name}_{f_id}")) 

            # Crea el payload con los datos del fotograma
            payload = {
                "frame_id": f_id,
                "video_name": video_name,
                "timestamp_seconds": ts
            }
            # Agrega el punto a la lista de puntos
            points.append(PointStruct(
                id=point_id,
                vector=vec,
                payload=payload
            ))

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    # Buscar vectores similares
    def search_similar(self,query_vector: List[float],limit: int = 5) -> List[SearchResult]:

        results = self.client.query_points(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=limit
        ).points
    
        search_results: List[SearchResult] = []
        for res in results:
            payload = res.payload or {}
            search_results.append(
                SearchResult(
                frame_id=payload.get("frame_id", ""),
                video_name=payload.get("video_name", ""),
                timestamp_seconds=payload.get("timestamp_seconds", 0.0),
                score=res.score
            ))
        return search_results
