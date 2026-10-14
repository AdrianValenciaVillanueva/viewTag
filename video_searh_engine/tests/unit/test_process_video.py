# Tests for ProcessVideoService
import pytest
from unittest.mock import Mock

from application.process_video import ProcessVideoService
from domain.models import Frame


class TestProcessVideoService:
    """Tests for the ProcessVideoService application service."""

    def test_execute_success(
        self,
        mock_frame_extractor_sync,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
        sample_frames,
        mock_normalized_vectors,
    ):
        """Test successful video processing flow."""
        service = ProcessVideoService(
            extractor=mock_frame_extractor_sync,
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        result = service.execute("/fake/path/video.mp4", "my_video")

        # Verify extractor called with correct path
        mock_frame_extractor_sync.extract.assert_called_once_with("/fake/path/video.mp4")

        # Verify embedder called with frames' images
        mock_siglip_embedder_sync.embed_image.assert_called_once()
        called_images = mock_siglip_embedder_sync.embed_image.call_args[0][0]
        assert len(called_images) == len(sample_frames)

        # Verify repository save_vectors called with correct data
        mock_qdrant_repo_sync.save_vectors.assert_called_once()
        call_kwargs = mock_qdrant_repo_sync.save_vectors.call_args[1]
        assert call_kwargs["video_name"] == "my_video"
        assert call_kwargs["frame_ids"] == [f.frame_id for f in sample_frames]
        assert call_kwargs["timestamps"] == [f.timestamp_seconds for f in sample_frames]
        assert call_kwargs["vectors"] == mock_normalized_vectors

        # Verify response
        assert result["video_name"] == "my_video"
        assert result["frames_processed"] == len(sample_frames)
        assert result["embeddings_saved"] == len(mock_normalized_vectors)

    def test_execute_uses_basename_when_video_name_none(
        self,
        mock_frame_extractor_sync,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
        sample_frames,
    ):
        """Test that video_name defaults to basename of video_path."""
        service = ProcessVideoService(
            extractor=mock_frame_extractor_sync,
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        result = service.execute("/path/to/some_video.mp4", None)

        call_kwargs = mock_qdrant_repo_sync.save_vectors.call_args[1]
        assert call_kwargs["video_name"] == "some_video"
        assert result["video_name"] == "some_video"

    def test_execute_empty_frames_returns_zero_counts(
        self,
        mock_frame_extractor_sync,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
    ):
        """Test that empty frame list returns zeros without calling embedder/repo."""
        mock_frame_extractor_sync.extract.return_value = []

        service = ProcessVideoService(
            extractor=mock_frame_extractor_sync,
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        result = service.execute("/fake/path/video.mp4")

        mock_siglip_embedder_sync.embed_image.assert_not_called()
        mock_qdrant_repo_sync.save_vectors.assert_not_called()

        assert result["video_name"] == "video"
        assert result["frames_processed"] == 0
        assert result["embeddings_saved"] == 0

    def test_execute_raises_value_error_on_empty_video_source(
        self,
        mock_frame_extractor_sync,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
    ):
        """Test that empty video_source raises ValueError."""
        service = ProcessVideoService(
            extractor=mock_frame_extractor_sync,
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        with pytest.raises(ValueError, match="video_source cannot be empty"):
            service.execute("")

        with pytest.raises(ValueError, match="video_source cannot be empty"):
            service.execute("   ")

    def test_execute_raises_value_error_on_empty_video_name(
        self,
        mock_frame_extractor_sync,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
    ):
        """Test that empty video_name (after basename extraction) raises ValueError."""
        service = ProcessVideoService(
            extractor=mock_frame_extractor_sync,
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        # Paths that result in empty basename after splitext (but non-empty video_source)
        with pytest.raises(ValueError, match="video_name no puede estar vacío"):
            service.execute("/")  # Root path -> empty basename

        with pytest.raises(ValueError, match="video_name no puede estar vacío"):
            service.execute("/path/")  # Directory path -> empty basename

    def test_execute_propagates_extractor_exception(
        self,
        mock_frame_extractor_sync,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
    ):
        """Test that extractor exceptions propagate."""
        mock_frame_extractor_sync.extract.side_effect = FileNotFoundError("Video not found")

        service = ProcessVideoService(
            extractor=mock_frame_extractor_sync,
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        with pytest.raises(FileNotFoundError, match="Video not found"):
            service.execute("/fake/path/video.mp4")

    def test_execute_propagates_embedder_exception(
        self,
        mock_frame_extractor_sync,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
        sample_frames,
    ):
        """Test that embedder exceptions propagate."""
        mock_siglip_embedder_sync.embed_image.side_effect = RuntimeError("CUDA OOM")

        service = ProcessVideoService(
            extractor=mock_frame_extractor_sync,
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        with pytest.raises(RuntimeError, match="CUDA OOM"):
            service.execute("/fake/path/video.mp4")

    def test_execute_propagates_repo_exception(
        self,
        mock_frame_extractor_sync,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
        sample_frames,
        mock_normalized_vectors,
    ):
        """Test that repository exceptions propagate."""
        mock_qdrant_repo_sync.save_vectors.side_effect = ConnectionError("Qdrant unavailable")

        service = ProcessVideoService(
            extractor=mock_frame_extractor_sync,
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        with pytest.raises(ConnectionError, match="Qdrant unavailable"):
            service.execute("/fake/path/video.mp4")