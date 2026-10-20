from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator

from observability.logging import configure_logging, get_logger

from api.routes import router
from application.process_video import ProcessVideoService
from application.search_video import SearchVideoService
from config.settings import settings
from infrastructure.database.qdrant_repo import QdrantVectorRepository
from infrastructure.ia.siglip_embedder import SigLIPEmbedder
from infrastructure.video.decord_extractor import DecordFrameExtractor


# Configurar logging estructurado al importar el módulo
configure_logging(json_logs=True, log_level="INFO")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    embedder = SigLIPEmbedder()
    repository = QdrantVectorRepository()
    extractor = DecordFrameExtractor()
    logger.info("startup", embedder_loaded=True, qdrant_connected=True)
    app.state.process = ProcessVideoService(extractor, embedder, repository)
    app.state.search = SearchVideoService(embedder, repository)
    yield
    # Shutdown graceful
    logger.info("shutdown_started")
    if hasattr(repository, 'client') and hasattr(repository.client, 'close'):
        try:
            repository.client.close()
            logger.info("qdrant_client_closed")
        except Exception as e:
            logger.warning("qdrant_close_failed", error=str(e))
    logger.info("shutdown_complete")


app = FastAPI(title="Video Search Engine", lifespan=lifespan)

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
        content=generate_latest(instrumentator.registry),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/")
async def read_root():
    return {"message": "Video Search Engine API"}


def main():
    """Entry point for the CLI command `viewtag-api`."""
    logger.info("starting_server", host=settings.api_host, port=settings.api_port)
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )


if __name__ == "__main__":
    main()