# Application service for searching video
from typing import List, Optional
from domain.interfaces.embedder import VectorEmbedderInterface
from domain.interfaces.repository import VectorRepositoryInterface, SearchPage
from domain.models import SearchResult


class SearchVideoService:
    def __init__(
        self,
        embedder: VectorEmbedderInterface,
        repository: VectorRepositoryInterface
    ):
        self.embedder = embedder
        self.repository = repository

    def execute(
        self,
        query: str,
        limit: int = 5,
        offset: int = 0,
        video_name: Optional[str] = None
    ) -> SearchPage:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if limit < 1 or limit > 50:
            raise ValueError("limit debe estar entre 1 y 50")

        if offset < 0:
            raise ValueError("offset debe ser >= 0")

        # Convert the query text into a vector representation
        query_vector = self.embedder.embed_text(query.strip())
        # Search for similar videos based on the query vector
        return self.repository.search_similar(
            query_vector=query_vector,
            limit=limit,
            offset=offset,
            video_name=video_name
        )

