# Concrete implementation for video extraction using Decord
import io
from typing import List
from decord import VideoReader, cpu
from PIL import Image

from src.domain.models import Frame #video_searh_engine\src\domain\models.py
from src.domain.interfaces.extractor import FrameExtractorInterface


class DecordFrameExtractor(FrameExtractorInterface):
    """
    Estrategia para extraer frames solo con Decord
    """
    def __init__(self, fps_sample_rate: float = 1.0):
        """
        :param fps_sample_rate: frames a extraer por segundo
                                ej: 
                                1.0 = extraer 1 frame por segundo
                                0.5 = extraer 1 frame cada 2 segundos
        """
        self.fps_sample_rate = fps_sample_rate

    def extract(self, video_path: str) -> List[Frame]:
        # 1. Cargar el lector de video
        reader = VideoReader(video_path, ctx=cpu(0))
        
        native_fps = reader.get_avg_fps()
        total_frames = len(reader)

        # 2. Calcular los índices de los frames a extraer
        step = max(1, int(native_fps / self.fps_sample_rate))
        frame_indices = list(range(0, total_frames, step))

        # 3. Extraer los frames en batch
        batch_frames = reader.get_batch(frame_indices).asnumpy()

        extracted_frames: List[Frame] = []

        # 4. Pasar los datos al modelo Frame
        #check to avoid immediate compression to jpeg
        for idx, frame_arr in zip(frame_indices, batch_frames):
            timestamp = idx / native_fps

            image = Image.fromarray(frame_arr)
            # buffer = io.BytesIO()
            # image.save(buffer, format="JPEG")
            # image_bytes = buffer.getvalue()

            # Creación de la entidad Frame
            frame_entity = Frame(
                frame_id=f"frame_{idx}",
                timestamp_seconds=round(timestamp, 2),
                image=image
            )

            extracted_frames.append(frame_entity)

        return extracted_frames