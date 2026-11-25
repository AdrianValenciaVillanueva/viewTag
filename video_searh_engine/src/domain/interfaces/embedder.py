# Define abstract interfaces for embedding generation

from abc import ABC, abstractmethod

from PIL import Image


class VectorEmbedderInterface(ABC):
    # contrato para modelo de ia que requieran embedder multimodal
    @abstractmethod
    async def embed_image(self, images: list[Image.Image]) -> list[list[float]]:
        """convierte los bytes de una imagen en una lista de numeros embed"""

        pass

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """convierte el texto de busqueda en un numero en el mismo espacio vectorial"""

        pass
