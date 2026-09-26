import pytest
import pytest_asyncio
from domain.models import Frame, SearchResult


class TestFixtures:
    """Test that conftest fixtures work correctly."""

    def test_sample_jpeg_bytes(self, sample_jpeg_bytes):
        assert isinstance(sample_jpeg_bytes, bytes)
        assert len(sample_jpeg_bytes) > 0

    def test_sample_frames(self, sample_frames):
        assert len(sample_frames) == 3
        for i, frame in enumerate(sample_frames):
            assert isinstance(frame, Frame)
            assert frame.frame_id == f"frame_{i}"
            assert frame.timestamp_seconds == float(i)
            assert isinstance(frame.image_bytes, bytes)

    def test_sample_search_results(self, sample_search_results):
        assert len(sample_search_results) == 3
        for i, result in enumerate(sample_search_results):
            assert isinstance(result, SearchResult)
            assert result.frame_id == f"frame_{i}"
            assert result.video_name == "test_video"
            assert result.timestamp_seconds == float(i)

    def test_mock_image_batch(self, mock_image_batch):
        from PIL import Image
        assert len(mock_image_batch) == 3
        for img in mock_image_batch:
            assert isinstance(img, Image.Image)
            assert img.size == (224, 224)

    def test_mock_normalized_vectors(self, mock_normalized_vectors):
        assert len(mock_normalized_vectors) == 3
        for vec in mock_normalized_vectors:
            assert len(vec) == 768
            # Check normalization (L2 norm ≈ 1)
            norm = sum(x * x for x in vec) ** 0.5
            assert abs(norm - 1.0) < 1e-5

    def test_mock_query_vector(self, mock_query_vector):
        assert len(mock_query_vector) == 768
        norm = sum(x * x for x in mock_query_vector) ** 0.5
        assert abs(norm - 1.0) < 1e-5


class TestAsyncFixtures:
    """Test async fixtures."""

    @pytest.mark.asyncio
    async def test_mock_siglip_embedder(self, mock_siglip_embedder):
        """Test async SigLIP embedder mock."""
        vectors = await mock_siglip_embedder.embed_image([])
        assert len(vectors) == 3
        assert len(vectors[0]) == 768

        query_vec = await mock_siglip_embedder.embed_text("test query")
        assert len(query_vec) == 768

        single_vec = await mock_siglip_embedder.embed_image_single(None)
        assert len(single_vec) == 768

    @pytest.mark.asyncio
    async def test_mock_frame_extractor(self, mock_frame_extractor, sample_frames):
        """Test async frame extractor mock."""
        frames = await mock_frame_extractor.extract("dummy_path")
        assert len(frames) == 3
        assert frames[0].frame_id == "frame_0"

    @pytest.mark.asyncio
    async def test_mock_qdrant_repo(self, mock_qdrant_repo, sample_search_results):
        """Test async Qdrant repo mock."""
        await mock_qdrant_repo.save_vectors("test", ["f1"], [0.0], [[0.1]*768])
        mock_qdrant_repo.save_vectors.assert_called_once()

        results = await mock_qdrant_repo.search_similar([0.1]*768)
        assert len(results) == 3
        assert results[0].video_name == "test_video"


class TestSyncFixtures:
    """Test sync fixtures."""

    def test_mock_siglip_embedder_sync(self, mock_siglip_embedder_sync):
        vectors = mock_siglip_embedder_sync.embed_image([])
        assert len(vectors) == 3

        query_vec = mock_siglip_embedder_sync.embed_text("test")
        assert len(query_vec) == 768

    def test_mock_frame_extractor_sync(self, mock_frame_extractor_sync, sample_frames):
        frames = mock_frame_extractor_sync.extract("dummy")
        assert len(frames) == 3

    def test_mock_qdrant_repo_sync(self, mock_qdrant_repo_sync, sample_search_results):
        mock_qdrant_repo_sync.save_vectors("test", ["f1"], [0.0], [[0.1]*768])
        mock_qdrant_repo_sync.save_vectors.assert_called_once()

        results = mock_qdrant_repo_sync.search_similar([0.1]*768)
        assert len(results) == 3