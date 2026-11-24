import hashlib
import json

import redis

from config.settings import settings
from domain.interfaces.cache import EmbeddingCacheInterface


class EmbeddingCache(EmbeddingCacheInterface):
    def __init__(self, redis_url: str | None = None, ttl: int | None = None):
        self.enabled = settings.embedding_cache_enabled and bool(redis_url or settings.redis_url)
        self.ttl = ttl or settings.embedding_cache_ttl_seconds
        url = redis_url or settings.redis_url
        self.client = None
        if url:
            self.client = redis.from_url(
                url,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )

    def _key(self, image_bytes: bytes, model_name: str) -> str:
        """Generate a cache key based on the image bytes and model name."""
        h = hashlib.sha256(image_bytes).hexdigest()
        return f"emb:{model_name}:{h}"

    def get(self, image_bytes: bytes, model_name: str) -> list[float] | None:
        """Retrieve the embedding from cache if available."""
        if not self.enabled or self.client is None:
            return None
        try:
            raw = self.client.get(self._key(image_bytes, model_name))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception:
            return None

    def get_many(self, images_bytes: list[bytes], model_name: str) -> list[list[float] | None]:
        if not self.enabled or self.client is None or not images_bytes:
            return [None] * len(images_bytes)
        try:
            keys = [self._key(b, model_name) for b in images_bytes]
            raws = self.client.mget(keys)
            result: list[list[float] | None] = []
            for raw in raws:
                if raw is None:
                    result.append(None)
                else:
                    result.append(json.loads(raw))
            return result
        except Exception:
            return [None] * len(images_bytes)

    def set(self, image_bytes: bytes, model_name: str, vector: list[float]) -> None:
        """Store the embedding in cache."""
        if not self.enabled or self.client is None:
            return
        try:
            self.client.setex(self._key(image_bytes, model_name), self.ttl, json.dumps(vector))
        except Exception:
            pass

    def close(self):
        """Close the Redis connection."""
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
