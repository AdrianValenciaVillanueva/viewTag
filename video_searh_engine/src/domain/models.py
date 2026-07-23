# Define data models here
from dataclasses import dataclass
from typing import List, Optional

#model for video frames 
@dataclass
class Frame:
    frame_id: str
    timestamp_seconds:float
    image_bite: float
    path: Optional[str]

#class para retorno de resultado
@dataclass
class Search_Result:
    frame_id: str
    video_name: str
    timestamp_seconds: float
    score: float #similitud de la busqueda


