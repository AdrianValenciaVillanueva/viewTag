# Integration test configuration and fixtures
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
import cv2
from qdrant_client.models import Distance, VectorParams, PointStruct


class TestQdrantRepository:
    """Test-specific Qdrant repository using in-memory client."""

    def __init__(self, client, collection_name: str):
        self.client = client
        self.collection = collection_name
        self.upsert_batch_size = 256
        # Ensure collection exists
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE,
                ),
            )

    def save_vectors(self, video_name, frame_ids, timestamps, vectors):
        if not (len(frame_ids) == len(timestamps) == len(vectors)):
            raise ValueError("Longitudes no coinciden")
        if not vectors:
            return

        points = []
        for f_id, ts, vec in zip(frame_ids, timestamps, vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_name}_{f_id}"))
            payload = {
                "frame_id": f_id,
                "video_name": video_name,
                "timestamp_seconds": ts,
            }
            points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        for i in range(0, len(points), self.upsert_batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i : i + self.upsert_batch_size],
                wait=True,
            )

    def search_similar(self, query_vector, limit=5, offset=0, video_name=None):
        from domain.interfaces.repository import SearchPage
        from domain.models import SearchResult
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Build optional filter
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

        # Get total count
        count_result = self.client.count(
            collection_name=self.collection,
            count_filter=query_filter,
            exact=True,
        )
        total = count_result.count

        # Query with pagination
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            offset=offset,
        ).points

        search_results = []
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


@pytest.fixture(scope="session")
def qdrant_in_memory():
    """Provide an in-memory Qdrant client for integration tests.

    Uses Qdrant's ':memory:' mode - no Docker required, fast, isolated.
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:", timeout=30)
    yield client
    client.close()


@pytest.fixture
def qdrant_test_repo(qdrant_in_memory, request):
    """Create a test repository using the in-memory Qdrant client.

    Each test gets a unique collection name based on the test function name.
    """
    # Unique collection per test
    collection_name = f"test_{request.node.name}"
    return TestQdrantRepository(qdrant_in_memory, collection_name)


@pytest.fixture
def synthetic_video_path():
    """Create a synthetic test video and return its path.

    Creates a 3-second video at 10 FPS with distinct colored frames:
    - Second 0-1: Red frames
    - Second 1-2: Green frames
    - Second 2-3: Blue frames
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name

    try:
        fps = 10
        duration_seconds = 3
        width, height = 320, 240
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for i in range(fps * duration_seconds):
            second = i // fps
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            if second == 0:
                frame[:, :] = (0, 0, 255)  # Red (BGR)
            elif second == 1:
                frame[:, :] = (0, 255, 0)  # Green
            else:
                frame[:, :] = (255, 0, 0)  # Blue

            cv2.putText(
                frame,
                f"Frame {i} (t={i/fps:.1f}s)",
                (20, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            writer.write(frame)

        writer.release()
        yield video_path

    finally:
        if os.path.exists(video_path):
            # Ensure any readers are closed
            import gc
            gc.collect()
            try:
                os.unlink(video_path)
            except PermissionError:
                pass  # Windows file locking, ignore


@pytest.fixture
def synthetic_video_path_short():
    """Create a shorter synthetic video (1 second, 2 FPS = 2 frames)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name

    try:
        fps = 2
        duration_seconds = 1
        width, height = 224, 224
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for i in range(fps * duration_seconds):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (0, 0, 255) if i % 2 == 0 else (255, 0, 0)
            writer.write(frame)

        writer.release()
        yield video_path

    finally:
        if os.path.exists(video_path):
            import gc
            gc.collect()
            try:
                os.unlink(video_path)
            except PermissionError:
                pass