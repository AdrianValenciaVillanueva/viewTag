# Tests for SearchVideoService
import pytest
from unittest.mock import Mock

from application.search_video import SearchVideoService
from domain.models import SearchResult


class TestSearchVideoService:
    """Tests for the SearchVideoService application service."""

    def test_execute_success(
        self,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
        mock_query_vector,
        sample_search_results,
    ):
        """Test successful search flow."""
        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        results = service.execute("buscar código", limit=3)

        # Verify embedder called with stripped query
        mock_siglip_embedder_sync.embed_text.assert_called_once_with("buscar código")

        # Verify repository search_similar called with query vector and limit
        mock_qdrant_repo_sync.search_similar.assert_called_once_with(
            query_vector=mock_query_vector, limit=3
        )

        # Verify results returned
        assert results == sample_search_results
        assert len(results) == 3
        assert results[0].frame_id == "frame_0"
        assert results[0].video_name == "test_video"
        assert results[0].score == 0.9

    def test_execute_strips_whitespace_from_query(
        self,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
        mock_query_vector,
        sample_search_results,
    ):
        """Test that query whitespace is stripped."""
        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        service.execute("  código  ", limit=5)

        mock_siglip_embedder_sync.embed_text.assert_called_once_with("código")

    def test_execute_uses_default_limit(
        self,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
        mock_query_vector,
        sample_search_results,
    ):
        """Test that default limit is used when not specified."""
        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        service.execute("código")

        mock_qdrant_repo_sync.search_similar.assert_called_once_with(
            query_vector=mock_query_vector, limit=5  # default from service
        )

    def test_execute_raises_value_error_on_empty_query(
        self,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
    ):
        """Test that empty query raises ValueError."""
        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        with pytest.raises(ValueError, match="query cannot be empty"):
            service.execute("")

        with pytest.raises(ValueError, match="query cannot be empty"):
            service.execute("   ")

        with pytest.raises(ValueError, match="query cannot be empty"):
            service.execute("\t\n")

    def test_execute_raises_value_error_on_invalid_limit_low(
        self,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
    ):
        """Test that limit < 1 raises ValueError."""
        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        with pytest.raises(ValueError, match="limit debe estar entre 1 y 50"):
            service.execute("código", limit=0)

        with pytest.raises(ValueError, match="limit debe estar entre 1 y 50"):
            service.execute("código", limit=-5)

    def test_execute_raises_value_error_on_invalid_limit_high(
        self,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
    ):
        """Test that limit > 50 raises ValueError."""
        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        with pytest.raises(ValueError, match="limit debe estar entre 1 y 50"):
            service.execute("código", limit=51)

        with pytest.raises(ValueError, match="limit debe estar entre 1 y 50"):
            service.execute("código", limit=100)

    def test_execute_accepts_boundary_limits(
        self,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
        mock_query_vector,
        sample_search_results,
    ):
        """Test that limit=1 and limit=50 are accepted."""
        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        # limit=1
        service.execute("código", limit=1)
        mock_qdrant_repo_sync.search_similar.assert_called_with(
            query_vector=mock_query_vector, limit=1
        )

        # limit=50
        service.execute("código", limit=50)
        mock_qdrant_repo_sync.search_similar.assert_called_with(
            query_vector=mock_query_vector, limit=50
        )

    def test_execute_propagates_embedder_exception(
        self,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
    ):
        """Test that embedder exceptions propagate."""
        mock_siglip_embedder_sync.embed_text.side_effect = RuntimeError("Model error")

        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        with pytest.raises(RuntimeError, match="Model error"):
            service.execute("código")

    def test_execute_propagates_repo_exception(
        self,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
        mock_query_vector,
    ):
        """Test that repository exceptions propagate."""
        mock_qdrant_repo_sync.search_similar.side_effect = ConnectionError("Qdrant down")

        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        with pytest.raises(ConnectionError, match="Qdrant down"):
            service.execute("código")

    def test_execute_returns_empty_list_when_no_results(
        self,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
        mock_query_vector,
    ):
        """Test that empty results from repo returns empty list."""
        mock_qdrant_repo_sync.search_similar.return_value = []

        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        results = service.execute("código que no existe")

        assert results == []
        assert isinstance(results, list)