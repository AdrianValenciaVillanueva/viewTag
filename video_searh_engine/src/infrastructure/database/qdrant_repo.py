# Concrete implementation for data persistence using Qdrant
import uuid
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)

# modelo e interfaz
from domain.interfaces.repository import VectorRepositoryInterface, SearchPage
from domain.models import SearchResult
from config.settings import settings


class QdrantVectorRepository(VectorRepositoryInterface):
    """Implementación Qdrant local con configuración centralizada."""

    def __init__(
        self,
        storage_path: str | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
    ):
        # Usar settings como valores por defecto; permite override en tests
        self.storage_path = storage_path or settings.qdrant_storage_path
        self.collection = collection_name or settings.qdrant_collection_name
        self.vector_size = vector_size or settings.qdrant_vector_size
        self.upsert_batch_size = settings.upsert_batch_size

        # Inicializar la base de datos local
        self.client = QdrantClient(path=self.storage_path)
        self._ensure_collection(self.vector_size)

    # Asegurar que la colección exista
    def _ensure_collection(self, vector_size: int) -> None:
        """Crea la colección local si aún no existe."""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

    # Guardar vectores en la base de datos
    def save_vectors(
        self,
        video_name: str,
        frame_ids: List[str],
        timestamps: List[float],
        vectors: List[List[float]]
    ) -> None:
        if not (len(frame_ids) == len(timestamps) == len(vectors)):
            raise ValueError(
                f"Longitudes no coinciden: "
                f"frame_ids={len(frame_ids)}, "
                f"timestamps={len(timestamps)}, "
                f"vectors={len(vectors)}"
            )
        if not vectors:
            return

        points = []

        for f_id, ts, vec in zip(frame_ids, timestamps, vectors):
            # Genera un ID único para cada punto
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_name}_{f_id}"))

            # Crea el payload con los datos del fotograma
            payload = {
                "frame_id": f_id,
                "video_name": video_name,
                "timestamp_seconds": ts
            }
            # Agrega el punto a la lista de puntos
            points.append(PointStruct(
                id=point_id,
                vector=vec,
                payload=payload
            ))

        for i in range(0, len(points), self.upsert_batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i : i + self.upsert_batch_size],
                wait=True,
            )

    # Buscar vectores similares con paginación y filtro opcional
    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 5,
        offset: int = 0,
        video_name: Optional[str] = None
    ) -> SearchPage:
        if not query_vector:
            raise ValueError("query_vector no puede estar vacío")
        if limit < 1:
            raise ValueError("limit debe ser >= 1")
        if offset < 0:
            raise ValueError("offset debe ser >= 0")

        # Construir filtro opcional por video_name
        query_filter = None
        if video_name:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="video_name",
                        match=MatchValue(value=video_name)
                    )
                ]
            )

        # Obtener total de coincidencias (para paginación)
        # Usamos count con el mismo filtro
        count_result = self.client.count(
            collection_name=self.collection,
            count_filter=query_filter,
            exact=True,
        )
        total = count_result.count

        # Buscar con paginación
        # Qdrant usa offset + limit
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            offset=offset,
        ).points

        search_results: List[SearchResult] = []
        for res in results:
            payload = res.payload or {}
            search_results.append(
                SearchResult(
                    frame_id=payload.get("frame_id", ""),
                    video_name=payload.get("video_name", ""),
                    timestamp_seconds=payload.get("timestamp_seconds", 0.0),
                    score=float(res.score) if res.score is not None else 0.0,
                )
            )

        return SearchPage(
            results=search_results,
            total=total,
            limit=limit,
            offset=offset,
        )
