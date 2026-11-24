import fakeredis
import pytest

from config.settings import settings
from infrastructure.cache.redis_cache import EmbeddingCache


@pytest.fixture
def fake_cache():
    settings.embedding_cache_enabled = True
    settings.redis_url = "redis://fake"
    cache = EmbeddingCache()
    cache.client = fakeredis.FakeRedis(decode_responses=False)
    yield cache
    settings.embedding_cache_enabled = True
    settings.redis_url = ""


def test_cache_miss_returns_none(fake_cache):
    assert fake_cache.get(b"fake-image", "model") is None


def test_cache_set_then_get_returns_vector(fake_cache):
    img = b"fake-image"
    vec = [0.1, 0.2, 0.3]
    fake_cache.set(img, "model", vec)
    assert fake_cache.get(img, "model") == vec


def test_cache_different_model_is_miss(fake_cache):
    img = b"fake-image"
    vec = [0.1, 0.2, 0.3]
    fake_cache.set(img, "model-a", vec)
    assert fake_cache.get(img, "model-b") is None


def test_cache_disabled_returns_none():
    settings.embedding_cache_enabled = False
    settings.redis_url = ""
    cache = EmbeddingCache()
    assert cache.get(b"img", "model") is None
    cache.set(b"img", "model", [1.0])
    assert cache.get(b"img", "model") is None
    settings.embedding_cache_enabled = True


def test_cache_close_does_not_raise(fake_cache):
    fake_cache.close()  # no debe lanzar
