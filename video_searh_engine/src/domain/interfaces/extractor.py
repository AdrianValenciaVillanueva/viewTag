# Define abstract interfaces for data extraction
from abc import ABC, abstractmethod

from domain.models import Frame


class FrameExtractorInterface(ABC):
    # contrato a cumplir
    @abstractmethod
    async def extract(self, video_path: str) -> list[Frame]:
        """
        extraer fotogramas de un archivo de video

        :param video_path: Ruta al archivo de video
        :return: Lista de objetos Frame

        """

        pass
