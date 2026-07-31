# Define data models here
from dataclasses import dataclass
from typing import List, Optional
from PIL import Image

#model for video frames 
@dataclass
class Frame:
    frame_id: str
    timestamp_seconds:float
    image_bytes: Image.Image #bytes
    path: Optional[str] = None

#class para retorno de resultado
@dataclass
class SearchResult:
    frame_id: str
    video_name: str
    timestamp_seconds: float
    score: float #similitud de la busqueda


