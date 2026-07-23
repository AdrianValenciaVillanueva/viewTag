# Define abstract interfaces for data extraction
from abc import ABC, abstractmethod
from typing import List
from src.domain.models import Frame

class FrameExtractorInterface(ABC):
    #contrato a cumplir
    @abstractmethod
    def extract(self,video_path:str) -> List[Frame]:
        """
        extraer fotogramas de un archivo de video

        :param video_path: Ruta al archivo de video
        :return: Lista de objetos Frame

        """

        pass
