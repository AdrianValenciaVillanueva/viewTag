# Implementación concreta de extracción de frames usando Decord.
import io
import os
from typing import List
from decord import VideoReader, cpu
from PIL import Image

from domain.models import Frame #video_searh_engine\src\domain\models.py
from domain.interfaces.extractor import FrameExtractorInterface

# Número máximo de frames pedidos a Decord en cada llamada a get_batch.
# Evita cargar el video completo en RAM de una sola vez (OOM).
_FRAME_BATCH_SIZE = 64


class DecordFrameExtractor(FrameExtractorInterface):
    """
    Estrategia para extraer frames solo con Decord.

    Devuelve 1 frame cada 'step' frames del video original, donde
    step = native_fps / fps_sample_rate, para aproximar los frames
    por segundo pedidos sin decodificar el video completo.
    """
    def __init__(self, fps_sample_rate: float = 1.0):
        """
        :param fps_sample_rate: frames a extraer por segundo
                                ej:
                                1.0 = extraer 1 frame por segundo
                                0.5 = extraer 1 frame cada 2 segundos
        """
        # El valor debe ser positivo: con 0 habría división por cero
        # y con negativo el 'step' saldría negativo y se ignoraría el parámetro.
        if fps_sample_rate <= 0:
            raise ValueError(f"fps_sample_rate debe ser > 0, recibido: {fps_sample_rate}")
        self.fps_sample_rate = fps_sample_rate

    def extract(self, video_path: str) -> List[Frame]:
        # Guardar la ruta ayuda a rastrear de qué video vino cada Frame.
        # Se valida antes de abrir Decord para dar un error claro en vez
        # del error interno y críptico del lector.
        if not video_path:
            raise ValueError("video_path no puede estar vacío")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video no encontrado: {video_path}")

        # 1. Abrir el lector de video en CPU.
        reader = VideoReader(video_path, ctx=cpu(0))

        # Algunos contenedores reportan fps 0/None; sin fps válido no se
        # puede calcular el 'step' ni el timestamp (división por cero).
        native_fps = reader.get_avg_fps()
        if native_fps is None or native_fps <= 0:
            raise ValueError(f"FPS inválido ({native_fps}) en video: {video_path}")

        total_frames = len(reader)

        # Un video sin frames no es error: se devuelve lista vacía para
        # que el pipeline (embedder/Qdrant) simplemente no tenga nada que hacer.
        # También se evita llamar a get_batch([]), que falla en Decord.
        if total_frames <= 0:
            return []

        # 2. Calcular los índices de los frames a extraer.
        # Ej: 30fps nativos con fps_sample_rate=1 -> step=30 -> 1 frame por segundo.
        step = max(1, int(native_fps / self.fps_sample_rate))
        frame_indices = list(range(0, total_frames, step))
        if not frame_indices:
            return []

        extracted_frames: List[Frame] = []

        # 3. Extraer por lotes para acotar el pico de RAM.
        # get_batch() devuelve un array numpy con todos los frames pedidos,
        # así que pedirlos todos de golpe (600+ frames 1080p) revienta memoria.
        for start in range(0, len(frame_indices), _FRAME_BATCH_SIZE):
            chunk_indices = frame_indices[start : start + _FRAME_BATCH_SIZE]
            batch_frames = reader.get_batch(chunk_indices).asnumpy()

            # 4. Convertir cada frame al modelo Frame.
            # Se comprime a JPEG porque Frame.image_bytes promete bytes
            # decodificables con Image.open (el raw de tobytes() no lo es
            # y además pierde width/height/mode para reconstruir).
            for idx, frame_arr in zip(chunk_indices, batch_frames):
                # El timestamp es la posición del frame dividida por los fps
                # reales del video, redondeado a centésimas para el payload.
                timestamp = idx / native_fps

                image = Image.fromarray(frame_arr)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=95)
                image_bytes = buffer.getvalue()

                # NOTA: frame_id no incluye el nombre del video a propósito
                # para mantener IDs estables; la unicidad entre videos la da
                # Qdrant con uuid5(f"{video_name}_{frame_id}").
                frame_entity = Frame(
                    frame_id=f"frame_{idx}",
                    timestamp_seconds=round(timestamp, 2),
                    image_bytes=image_bytes,  # JPEG comprimido
                    path=video_path,
                )

                extracted_frames.append(frame_entity)

        return extracted_frames
