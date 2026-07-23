# Define abstract interfaces for embedding generation

from abc import ABC, abstractmethod
from typing import List

class VectorEmbedderInterface(ABC):
    #contrato para modelo de ia que requieran embedder multimodal
    @abstractmethod
    def embed_image(self, image_bytes:bytes) -> List[float]:

        """convierte los bytes de una imagen en una lista de numeros embed"""

        pass

    @abstractmethod
    def embed_text(self, text:str) -> List[float]:

        """convierte el texto de busqueda en un numero en el mismo espacio vectorial"""

        pass
