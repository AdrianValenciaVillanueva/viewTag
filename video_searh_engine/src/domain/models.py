# Define data models here
import io
from dataclasses import dataclass

from PIL import Image


# model for video frames
@dataclass
class Frame:
    frame_id: str
    timestamp_seconds: float
    image_bytes: bytes  # JPEG comprimido
    path: str | None = None

    @property
    def image(self) -> Image.Image:  # lazy load para compatibilidad
        img = Image.open(io.BytesIO(self.image_bytes))
        img.load()
        return img


# class para retorno de resultado
@dataclass
class SearchResult:
    frame_id: str
    video_name: str
    timestamp_seconds: float
    score: float  # similitud de la busqueda
