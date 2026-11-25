# Integration tests for the full video search pipeline
import pytest

from application.process_video import ProcessVideoService
from application.search_video import SearchVideoService
from domain.interfaces.repository import SearchPage
from infrastructure.ia.siglip_embedder import SigLIPEmbedder
from infrastructure.video.decord_extractor import DecordFrameExtractor


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipeline:
    """Integration tests using real Qdrant (in-memory), real model, real video."""

    @pytest.mark.asyncio
    async def test_process_and_search_synthetic_video(self, qdrant_test_repo, synthetic_video_path):
        """Test full pipeline: process video → search for content."""
        repository = qdrant_test_repo
        embedder = SigLIPEmbedder()
        extractor = DecordFrameExtractor(fps_sample_rate=1.0)

        process_service = ProcessVideoService(extractor, embedder, repository)
        search_service = SearchVideoService(embedder, repository)

        result = await process_service.execute(synthetic_video_path, "test_video")

        assert result["video_name"] == "test_video"
        assert result["frames_processed"] == 3
        assert result["embeddings_saved"] == 3

        page = await search_service.execute("red", limit=3)

        assert isinstance(page, SearchPage)
        assert len(page.results) > 0
        assert all(r.video_name == "test_video" for r in page.results)
        assert page.results[0].timestamp_seconds < 1.5
        assert page.total >= len(page.results)
        assert page.limit == 3
        assert page.offset == 0

    @pytest.mark.asyncio
    async def test_search_returns_empty_for_unrelated_query(
        self, qdrant_test_repo, synthetic_video_path
    ):
        """Search for something not in video should return low scores."""
        repository = qdrant_test_repo
        embedder = SigLIPEmbedder()
        extractor = DecordFrameExtractor(fps_sample_rate=1.0)

        process_service = ProcessVideoService(extractor, embedder, repository)
        search_service = SearchVideoService(embedder, repository)

        await process_service.execute(synthetic_video_path, "test_video_2")

        page = await search_service.execute("unrelated abstract concept xyz", limit=5)

        assert isinstance(page, SearchPage)
        if page.results:
            assert all(r.score < 0.5 for r in page.results)

    @pytest.mark.asyncio
    async def test_reprocessing_same_video_overwrites(self, qdrant_test_repo, synthetic_video_path):
        """Processing the same video twice should overwrite, not duplicate."""
        repository = qdrant_test_repo
        embedder = SigLIPEmbedder()
        extractor = DecordFrameExtractor(fps_sample_rate=1.0)

        process_service = ProcessVideoService(extractor, embedder, repository)

        result1 = await process_service.execute(synthetic_video_path, "overwrite_test")
        assert result1["embeddings_saved"] == 3

        result2 = await process_service.execute(synthetic_video_path, "overwrite_test")
        assert result2["embeddings_saved"] == 3

        search_service = SearchVideoService(embedder, repository)
        page = await search_service.execute("red", limit=10)
        assert len(page.results) == 3
        assert page.total == 3

    @pytest.mark.asyncio
    async def test_different_videos_stored_separately(
        self, qdrant_test_repo, synthetic_video_path, synthetic_video_path_short
    ):
        """Two different videos should be stored and searchable independently."""
        repository = qdrant_test_repo
        embedder = SigLIPEmbedder()
        extractor = DecordFrameExtractor(fps_sample_rate=1.0)

        process_service = ProcessVideoService(extractor, embedder, repository)
        search_service = SearchVideoService(embedder, repository)

        await process_service.execute(synthetic_video_path, "video_a")
        await process_service.execute(synthetic_video_path_short, "video_b")

        page = await search_service.execute("red", limit=10)

        video_names = {r.video_name for r in page.results}
        assert "video_a" in video_names
        assert "video_b" in video_names
        assert page.total >= 2

    @pytest.mark.asyncio
    async def test_limit_parameter_respected(self, qdrant_test_repo, synthetic_video_path):
        """Search limit parameter should be respected."""
        repository = qdrant_test_repo
        embedder = SigLIPEmbedder()
        extractor = DecordFrameExtractor(fps_sample_rate=1.0)

        process_service = ProcessVideoService(extractor, embedder, repository)
        search_service = SearchVideoService(embedder, repository)

        await process_service.execute(synthetic_video_path, "limit_test")

        page = await search_service.execute("red", limit=1)
        assert len(page.results) == 1
        assert page.limit == 1

        page = await search_service.execute("red", limit=2)
        assert len(page.results) == 2
        assert page.limit == 2

    @pytest.mark.asyncio
    async def test_fps_sample_rate_affects_frame_count(
        self, qdrant_in_memory, synthetic_video_path
    ):
        """Different fps_sample_rate should extract different number of frames."""
        embedder = SigLIPEmbedder()

        from tests.integration.conftest import TestQdrantRepository

        repository1 = TestQdrantRepository(qdrant_in_memory, "fps_test_1")
        extractor_1fps = DecordFrameExtractor(fps_sample_rate=1.0)
        process_1fps = ProcessVideoService(extractor_1fps, embedder, repository1)
        result_1fps = await process_1fps.execute(synthetic_video_path, "fps_test_1")
        assert result_1fps["frames_processed"] == 3

        repository2 = TestQdrantRepository(qdrant_in_memory, "fps_test_half")
        extractor_half_fps = DecordFrameExtractor(fps_sample_rate=0.5)
        process_half_fps = ProcessVideoService(extractor_half_fps, embedder, repository2)
        result_half_fps = await process_half_fps.execute(synthetic_video_path, "fps_test_half")
        assert result_half_fps["frames_processed"] == 2

    @pytest.mark.asyncio
    async def test_batch_sizes_respected(self, qdrant_test_repo, synthetic_video_path):
        """Custom batch sizes should be respected during processing."""
        embedder = SigLIPEmbedder(image_batch_size=1)
        extractor = DecordFrameExtractor(fps_sample_rate=1.0, frame_batch_size=1)

        process_service = ProcessVideoService(extractor, embedder, qdrant_test_repo)
        result = await process_service.execute(synthetic_video_path, "batch_test")

        assert result["frames_processed"] == 3
        assert result["embeddings_saved"] == 3
