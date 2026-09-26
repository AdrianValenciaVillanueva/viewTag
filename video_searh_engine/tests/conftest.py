import io
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import pytest_asyncio
import torch
from PIL import Image

from domain.models import Frame, SearchResult
from infrastructure.database import qdrant_repo as qdrant_module

# Import infrastructure modules to make them patchable
from infrastructure.ia import siglip_embedder as siglip_module
from infrastructure.video import decord_extractor as decord_module


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Generate a small JPEG image as bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def sample_frames(sample_jpeg_bytes: bytes) -> list[Frame]:
    """Create sample Frame objects for testing."""
    return [
        Frame(frame_id=f"frame_{i}", timestamp_seconds=float(i), image_bytes=sample_jpeg_bytes)
        for i in range(3)
    ]


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Create sample SearchResult objects for testing."""
    return [
        SearchResult(
            frame_id=f"frame_{i}",
            video_name="test_video",
            timestamp_seconds=float(i),
            score=0.9 - i * 0.1,
        )
        for i in range(3)
    ]


@pytest.fixture
def mock_image_batch() -> list[Image.Image]:
    """Create a batch of PIL Images for testing."""
    return [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(3)
    ]


@pytest.fixture
def mock_normalized_vectors() -> list[list[float]]:
    """Create mock normalized vectors (768 dims)."""
    vectors = torch.randn(3, 768)
    vectors = vectors / vectors.norm(dim=1, keepdim=True)
    return vectors.cpu().tolist()


@pytest.fixture
def mock_query_vector() -> list[float]:
    """Create a mock normalized query vector."""
    vec = torch.randn(768)
    vec = vec / vec.norm()
    return vec.cpu().tolist()


# --- Async Mock Fixtures ---


@pytest_asyncio.fixture
async def mock_siglip_embedder(mock_normalized_vectors, mock_query_vector):
    """Async mock for SigLIPEmbedder."""
    with patch.object(siglip_module, "SigLIPEmbedder", autospec=True) as mock_class:
        mock_instance = AsyncMock()
        mock_instance.embed_image = AsyncMock(return_value=mock_normalized_vectors)
        mock_instance.embed_text = AsyncMock(return_value=mock_query_vector)
        mock_instance.embed_image_single = AsyncMock(return_value=mock_normalized_vectors[0])
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest_asyncio.fixture
async def mock_frame_extractor(sample_frames):
    """Async mock for DecordFrameExtractor."""
    with patch.object(decord_module, "DecordFrameExtractor", autospec=True) as mock_class:
        mock_instance = AsyncMock()
        mock_instance.extract = AsyncMock(return_value=sample_frames)
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest_asyncio.fixture
async def mock_qdrant_repo(sample_search_results):
    """Async mock for QdrantVectorRepository."""
    with patch.object(qdrant_module, "QdrantVectorRepository", autospec=True) as mock_class:
        mock_instance = AsyncMock()
        mock_instance.save_vectors = AsyncMock(return_value=None)
        mock_instance.search_similar = AsyncMock(return_value=sample_search_results)
        mock_class.return_value = mock_instance
        yield mock_instance


# --- Sync Mock Fixtures (for non-async tests) ---


@pytest.fixture
def mock_siglip_embedder_sync(mock_normalized_vectors, mock_query_vector):
    """Sync mock for SigLIPEmbedder."""
    with patch.object(siglip_module, "SigLIPEmbedder", autospec=True) as mock_class:
        mock_instance = Mock()
        mock_instance.embed_image = Mock(return_value=mock_normalized_vectors)
        mock_instance.embed_text = Mock(return_value=mock_query_vector)
        mock_instance.embed_image_single = Mock(return_value=mock_normalized_vectors[0])
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_frame_extractor_sync(sample_frames):
    """Sync mock for DecordFrameExtractor."""
    with patch.object(decord_module, "DecordFrameExtractor", autospec=True) as mock_class:
        mock_instance = Mock()
        mock_instance.extract = Mock(return_value=sample_frames)
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_qdrant_repo_sync(sample_search_results):
    """Sync mock for QdrantVectorRepository."""
    with patch.object(qdrant_module, "QdrantVectorRepository", autospec=True) as mock_class:
        mock_instance = Mock()
        mock_instance.save_vectors = Mock(return_value=None)
        mock_instance.search_similar = Mock(return_value=sample_search_results)
        mock_class.return_value = mock_instance
        yield mock_instance


# --- Pytest Configuration ---


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, mocked)")
    config.addinivalue_line("markers", "integration: Integration tests (real dependencies)")
    config.addinivalue_line("markers", "slow: Slow tests (model loading, real API calls)")


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton state between tests."""
    yield
