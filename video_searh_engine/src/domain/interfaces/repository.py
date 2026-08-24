# Define abstract interfaces for data persistence
from abc import ABC, abstractmethod
from typing import List
from video_searh_engine.src.domain.models import SearchResult

class VectorRepositoryInterface(ABC):

    @abstractmethod
    def save_vectors(
        self,
        video_name: str,
        frame_ids: List[str],
        timestamps: List[float],
        vectors: List[float]
    ) -> None:
        """guarda un conjunto de vectores y sus metadatos en la base de datos"""
        pass

    @abstractmethod
    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 5
    ) -> List[SearchResult]:
        """busca los vectores mas cercanos a un vector de consulta"""
        pass