# Define abstract interfaces for data persistence
from abc import ABC, abstractmethod
from dataclasses import dataclass

from domain.models import SearchResult


@dataclass
class SearchPage:
    """Paginated search results with metadata."""

    results: list[SearchResult]
    total: int
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        """Whether there are more results after this page."""
        return self.offset + self.limit < self.total


class VectorRepositoryInterface(ABC):
    @abstractmethod
    def save_vectors(
        self,
        video_name: str,
        frame_ids: list[str],
        timestamps: list[float],
        vectors: list[list[float]],
    ) -> None:
        """guarda un conjunto de vectores y sus metadatos en la base de datos"""
        pass

    @abstractmethod
    def search_similar(
        self,
        query_vector: list[float],
        limit: int = 5,
        offset: int = 0,
        video_name: str | None = None,
    ) -> SearchPage:
        """
        Busca los vectores más cercanos a un vector de consulta con paginación y filtro opcional.

        Args:
            query_vector: Vector de consulta normalizado
            limit: Máximo número de resultados a devolver
            offset: Número de resultados a saltar (para paginación)
            video_name: Si se especifica, filtra resultados a este video

        Returns:
            SearchPage con results, total, limit, offset
        """
        pass
