# Application service for searching video
from typing import List
from domain.interfaces.embedder import VectorEmbedderInterface
from domain.interfaces.repository import VectorRepositoryInterface
from domain.models import SearchResult
class SearchVideoService:
    def __init__(self, embedder: VectorEmbedderInterface, repository: VectorRepositoryInterface):
        self.embedder = embedder
        self.repository = repository

    def execute(self, query: str, limit: int = 5) -> List[SearchResult]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        
        if limit < 1 or limit > 50:
            raise ValueError("limit debe estar entre 1 y 50")

        # Convert the query text into a vector representation
        query_vector = self.embedder.embed_text(query.strip())
        # Search for similar videos based on the query vector
        return self.repository.search_similar(query_vector=query_vector, limit=limit)

