# Tests for SearchVideoService
import pytest
from unittest.mock import Mock

from application.search_video import SearchVideoService
from domain.interfaces.repository import SearchPage
from domain.models import SearchResult


def make_search_page(results, limit=5, offset=0, total=None):
    """Helper to create a SearchPage for test mocking."""
    if total is None:
        total = len(results)
    return SearchPage(
        results=results,
        total=total,
        limit=limit,
        offset=offset,
    )


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
        # Mock returns a SearchPage now
        mock_page = make_search_page(sample_search_results, limit=3, total=3)
        mock_qdrant_repo_sync.search_similar.return_value = mock_page

        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        page = service.execute("buscar código", limit=3)

        # Verify embedder called with stripped query
        mock_siglip_embedder_sync.embed_text.assert_called_once_with("buscar código")

        # Verify repository search_similar called with query vector, limit, offset, video_name
        mock_qdrant_repo_sync.search_similar.assert_called_once_with(
            query_vector=mock_query_vector, limit=3, offset=0, video_name=None
        )

        # Verify SearchPage returned
        assert isinstance(page, SearchPage)
        assert page.results == sample_search_results
        assert page.total == 3
        assert page.limit == 3
        assert page.offset == 0
        assert page.has_more is False

    def test_execute_strips_whitespace_from_query(
        self,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
        mock_query_vector,
        sample_search_results,
    ):
        """Test that query whitespace is stripped."""
        mock_page = make_search_page(sample_search_results, limit=5, total=3)
        mock_qdrant_repo_sync.search_similar.return_value = mock_page

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
        mock_page = make_search_page(sample_search_results, limit=5, total=3)
        mock_qdrant_repo_sync.search_similar.return_value = mock_page

        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        service.execute("código")

        mock_qdrant_repo_sync.search_similar.assert_called_once_with(
            query_vector=mock_query_vector, limit=5, offset=0, video_name=None
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

    def test_execute_raises_value_error_on_invalid_offset(
        self,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
    ):
        """Test that negative offset raises ValueError."""
        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        with pytest.raises(ValueError, match="offset debe ser >= 0"):
            service.execute("código", offset=-1)

    def test_execute_accepts_boundary_limits(
        self,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
        mock_query_vector,
        sample_search_results,
    ):
        """Test that limit=1 and limit=50 are accepted."""
        mock_page = make_search_page(sample_search_results, limit=1, total=3)
        mock_qdrant_repo_sync.search_similar.return_value = mock_page

        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        # limit=1
        service.execute("código", limit=1)
        mock_qdrant_repo_sync.search_similar.assert_called_with(
            query_vector=mock_query_vector, limit=1, offset=0, video_name=None
        )

        # limit=50
        mock_page = make_search_page(sample_search_results, limit=50, total=3)
        mock_qdrant_repo_sync.search_similar.return_value = mock_page
        service.execute("código", limit=50)
        mock_qdrant_repo_sync.search_similar.assert_called_with(
            query_vector=mock_query_vector, limit=50, offset=0, video_name=None
        )

    def test_execute_passes_offset_and_video_name(
        self,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
        mock_query_vector,
        sample_search_results,
    ):
        """Test that offset and video_name are passed to repository."""
        mock_page = make_search_page(sample_search_results, limit=3, offset=2, total=10)
        mock_qdrant_repo_sync.search_similar.return_value = mock_page

        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        page = service.execute("código", limit=3, offset=2, video_name="my_video")

        mock_qdrant_repo_sync.search_similar.assert_called_once_with(
            query_vector=mock_query_vector, limit=3, offset=2, video_name="my_video"
        )
        assert page.offset == 2
        assert page.total == 10
        assert page.has_more is True  # 2 + 3 < 10

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

    def test_execute_returns_empty_page_when_no_results(
        self,
        mock_siglip_embedder_sync,
        mock_qdrant_repo_sync,
        mock_query_vector,
    ):
        """Test that empty results from repo returns empty SearchPage."""
        mock_page = make_search_page([], limit=5, total=0)
        mock_qdrant_repo_sync.search_similar.return_value = mock_page

        service = SearchVideoService(
            embedder=mock_siglip_embedder_sync,
            repository=mock_qdrant_repo_sync,
        )

        page = service.execute("código que no existe")

        assert isinstance(page, SearchPage)
        assert page.results == []
        assert page.total == 0
        assert page.has_more is False