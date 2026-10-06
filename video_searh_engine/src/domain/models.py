# Define data models here
from dataclasses import dataclass
import io
from typing import List, Optional
from PIL import Image

#model for video frames 
@dataclass
class Frame:
    frame_id: str
    timestamp_seconds: float
    image_bytes: bytes  # JPEG comprimido
    path: Optional[str] = None
    
    @property
    def image(self) -> Image.Image:  # lazy load para compatibilidad
        img = Image.open(io.BytesIO(self.image_bytes))
        img.load()
        return img

#class para retorno de resultado
@dataclass
class SearchResult:
    frame_id: str
    video_name: str
    timestamp_seconds: float
    score: float #similitud de la busqueda


