# Tests for ProcessVideoService

import pytest

from application.process_video import ProcessVideoService


class TestProcessVideoService:
    """Tests for the ProcessVideoService application service."""

    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        mock_frame_extractor,
        mock_siglip_embedder,
        mock_qdrant_repo,
        sample_frames,
        mock_normalized_vectors,
    ):
        """Test successful video processing flow."""
        service = ProcessVideoService(
            extractor=mock_frame_extractor,
            embedder=mock_siglip_embedder,
            repository=mock_qdrant_repo,
        )

        result = await service.execute("/fake/path/video.mp4", "my_video")

        mock_frame_extractor.extract.assert_awaited_once_with("/fake/path/video.mp4")

        mock_siglip_embedder.embed_image.assert_awaited_once()
        called_images = mock_siglip_embedder.embed_image.call_args[0][0]
        assert len(called_images) == len(sample_frames)

        mock_qdrant_repo.save_vectors.assert_awaited_once()
        call_kwargs = mock_qdrant_repo.save_vectors.call_args[1]
        assert call_kwargs["video_name"] == "my_video"
        assert call_kwargs["frame_ids"] == [f.frame_id for f in sample_frames]
        assert call_kwargs["timestamps"] == [f.timestamp_seconds for f in sample_frames]
        assert call_kwargs["vectors"] == mock_normalized_vectors

        assert result["video_name"] == "my_video"
        assert result["frames_processed"] == len(sample_frames)
        assert result["embeddings_saved"] == len(mock_normalized_vectors)

    @pytest.mark.asyncio
    async def test_execute_uses_basename_when_video_name_none(
        self, mock_frame_extractor, mock_siglip_embedder, mock_qdrant_repo, sample_frames
    ):
        """Test that video_name defaults to basename of video_path."""
        service = ProcessVideoService(
            extractor=mock_frame_extractor,
            embedder=mock_siglip_embedder,
            repository=mock_qdrant_repo,
        )

        result = await service.execute("/path/to/some_video.mp4", None)

        call_kwargs = mock_qdrant_repo.save_vectors.call_args[1]
        assert call_kwargs["video_name"] == "some_video"
        assert result["video_name"] == "some_video"

    @pytest.mark.asyncio
    async def test_execute_empty_frames_returns_zero_counts(
        self, mock_frame_extractor, mock_siglip_embedder, mock_qdrant_repo
    ):
        """Test that empty frame list returns zeros without calling embedder/repo."""
        mock_frame_extractor.extract.return_value = []

        service = ProcessVideoService(
            extractor=mock_frame_extractor,
            embedder=mock_siglip_embedder,
            repository=mock_qdrant_repo,
        )

        result = await service.execute("/fake/path/video.mp4")

        mock_siglip_embedder.embed_image.assert_not_awaited()
        mock_qdrant_repo.save_vectors.assert_not_awaited()

        assert result["video_name"] == "video"
        assert result["frames_processed"] == 0
        assert result["embeddings_saved"] == 0

    @pytest.mark.asyncio
    async def test_execute_raises_value_error_on_empty_video_source(
        self, mock_frame_extractor, mock_siglip_embedder, mock_qdrant_repo
    ):
        """Test that empty video_source raises ValueError."""
        service = ProcessVideoService(
            extractor=mock_frame_extractor,
            embedder=mock_siglip_embedder,
            repository=mock_qdrant_repo,
        )

        with pytest.raises(ValueError, match="video_source cannot be empty"):
            await service.execute("")

        with pytest.raises(ValueError, match="video_source cannot be empty"):
            await service.execute("   ")

    @pytest.mark.asyncio
    async def test_execute_raises_value_error_on_empty_video_name(
        self, mock_frame_extractor, mock_siglip_embedder, mock_qdrant_repo
    ):
        """Test that empty video_name (after basename extraction) raises ValueError."""
        service = ProcessVideoService(
            extractor=mock_frame_extractor,
            embedder=mock_siglip_embedder,
            repository=mock_qdrant_repo,
        )

        with pytest.raises(ValueError, match="video_name no puede estar vacío"):
            await service.execute("/")

        with pytest.raises(ValueError, match="video_name no puede estar vacío"):
            await service.execute("/path/")

    @pytest.mark.asyncio
    async def test_execute_propagates_extractor_exception(
        self, mock_frame_extractor, mock_siglip_embedder, mock_qdrant_repo
    ):
        """Test that extractor exceptions propagate."""
        mock_frame_extractor.extract.side_effect = FileNotFoundError("Video not found")

        service = ProcessVideoService(
            extractor=mock_frame_extractor,
            embedder=mock_siglip_embedder,
            repository=mock_qdrant_repo,
        )

        with pytest.raises(FileNotFoundError, match="Video not found"):
            await service.execute("/fake/path/video.mp4")

    @pytest.mark.asyncio
    async def test_execute_propagates_embedder_exception(
        self, mock_frame_extractor, mock_siglip_embedder, mock_qdrant_repo, sample_frames
    ):
        """Test that embedder exceptions propagate."""
        mock_siglip_embedder.embed_image.side_effect = RuntimeError("CUDA OOM")

        service = ProcessVideoService(
            extractor=mock_frame_extractor,
            embedder=mock_siglip_embedder,
            repository=mock_qdrant_repo,
        )

        with pytest.raises(RuntimeError, match="CUDA OOM"):
            await service.execute("/fake/path/video.mp4")

    @pytest.mark.asyncio
    async def test_execute_propagates_repo_exception(
        self,
        mock_frame_extractor,
        mock_siglip_embedder,
        mock_qdrant_repo,
        sample_frames,
        mock_normalized_vectors,
    ):
        """Test that repository exceptions propagate."""
        mock_qdrant_repo.save_vectors.side_effect = ConnectionError("Qdrant unavailable")

        service = ProcessVideoService(
            extractor=mock_frame_extractor,
            embedder=mock_siglip_embedder,
            repository=mock_qdrant_repo,
        )

        with pytest.raises(ConnectionError, match="Qdrant unavailable"):
            await service.execute("/fake/path/video.mp4")
