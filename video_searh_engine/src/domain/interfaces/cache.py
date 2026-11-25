from abc import ABC, abstractmethod


class EmbeddingCacheInterface(ABC):
    @abstractmethod
    async def get(self, image_bytes: bytes, model_name: str) -> list[float] | None:
        """Retrieve the embedding from cache if available."""
        pass

    @abstractmethod
    async def get_many(
        self, images_bytes: list[bytes], model_name: str
    ) -> list[list[float] | None]:
        """Retrieve multiple embeddings in one round trip."""
        pass

    @abstractmethod
    async def set(self, image_bytes: bytes, model_name: str, vector: list[float]) -> None:
        """Store the embedding in cache."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the cache connection."""
        pass
