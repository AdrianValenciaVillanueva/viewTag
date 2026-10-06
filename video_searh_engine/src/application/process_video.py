# Application service for processing video
import os
from typing import Dict, Optional
from domain.interfaces.extractor import FrameExtractorInterface
from domain.interfaces.embedder import VectorEmbedderInterface
from domain.interfaces.repository import VectorRepositoryInterface
class ProcessVideoService:
    def __init__(self, extractor: FrameExtractorInterface, embedder: VectorEmbedderInterface, repository: VectorRepositoryInterface):
        self.extractor = extractor
        self.embedder = embedder
        self.repository = repository

    def execute(self, video_source: str, video_name: Optional[str] = None) -> Dict[str, object]:
        # Logic to extract, embed, and save video data
        if not video_source:
            raise ValueError("video_source cannot be empty")

        # Clave estable para Qdrant (uuid5): el basename evita duplicar
        # puntos al reprocesar el mismo video desde otro CWD.
        resolved_name = video_name or os.path.splitext(os.path.basename(video_source))[0]
        if not resolved_name:
            raise ValueError("video_name no puede estar vacío")

        frames = self.extractor.extract(video_source)
        if not frames:
            return ({"video_name": resolved_name, "frames_processed": 0, "embeddings_saved": 0})

        images = [f.image for f in frames]
        embeddings = self.embedder.embed_image(images)
        
        self.repository.save_vectors(video_name=resolved_name, frame_ids=[f.frame_id for f in frames], timestamps=[f.timestamp_seconds for f in frames], vectors=embeddings)
        return ({"video_name": resolved_name, "frames_processed": len(frames), "embeddings_saved": len(embeddings)})
