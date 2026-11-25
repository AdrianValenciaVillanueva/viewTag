from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api.rate_limit import setup_rate_limiting
from api.routes import router
from application.process_video import ProcessVideoService
from application.search_video import SearchVideoService
from config.settings import settings
from infrastructure.cache.redis_cache import EmbeddingCache
from infrastructure.database.qdrant_repo import QdrantVectorRepository
from infrastructure.ia.siglip_embedder import SigLIPEmbedder
from infrastructure.video.decord_extractor import DecordFrameExtractor
from observability.logging import configure_logging, get_logger

# Configurar logging estructurado al importar el módulo
configure_logging(json_logs=True, log_level="INFO")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cache = EmbeddingCache() if settings.embedding_cache_enabled and settings.redis_url else None
    embedder = SigLIPEmbedder(cache=cache)
    repository = QdrantVectorRepository()
    extractor = DecordFrameExtractor()
    logger.info(
        "startup", embedder_loaded=True, qdrant_connected=True, cache_enabled=cache is not None
    )
    app.state.process = ProcessVideoService(extractor, embedder, repository)
    app.state.search = SearchVideoService(embedder, repository)
    app.state.cache = cache
    yield
    # Shutdown graceful
    logger.info("shutdown_started")
    if cache is not None:
        try:
            await cache.close()
            logger.info("embedding_cache_closed")
        except Exception as e:
            logger.warning("embedding_cache_close_failed", error=str(e))
    if hasattr(repository, "client") and hasattr(repository.client, "close"):
        try:
            repository.client.close()
            logger.info("qdrant_client_closed")
        except Exception as e:
            logger.warning("qdrant_close_failed", error=str(e))
    logger.info("shutdown_complete")


app = FastAPI(title="Video Search Engine", lifespan=lifespan)

# CORS (solo si hay orígenes explícitos; si no, same-origin)
allowed_origins = [
    origin.strip() for origin in settings.cors_allowed_origins.split(",") if origin.strip()
]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Límite de tamaño de request
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_request_size_mb * 1024 * 1024:
            return Response(status_code=413, content="Request body too large")
        return await call_next(request)


app.add_middleware(RequestSizeLimitMiddleware)


# Rate limiting
setup_rate_limiting(app)

# 1. Workaround: agregar rutas del router manualmente (include_router no funciona)
for route in router.routes:
    app.router.routes.append(route)

# 2. Prometheus metrics - ANTES de endpoints que lo usan
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/health", "/metrics"],
).instrument(app)


# 2. Endpoints adicionales
@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(
        content=generate_latest(instrumentator.registry), media_type=CONTENT_TYPE_LATEST
    )


@app.get("/")
async def read_root():
    return {"message": "Video Search Engine API"}


def main():
    """Entry point for the CLI command `viewtag-api`."""
    logger.info("starting_server", host=settings.api_host, port=settings.api_port)
    import uvicorn

    uvicorn.run(
        "api.main:app", host=settings.api_host, port=settings.api_port, reload=settings.api_reload
    )


if __name__ == "__main__":
    main()
