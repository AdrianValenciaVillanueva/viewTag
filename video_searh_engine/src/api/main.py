from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.routes import router
from application.process_video import ProcessVideoService
from application.search_video import SearchVideoService
from infrastructure.database.qdrant_repo import QdrantVectorRepository
from infrastructure.ia.siglip_embedder import SigLIPEmbedder
from infrastructure.video.decord_extractor import DecordFrameExtractor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # los singletons caros se crean una vez al arrancar, no por request:
    embedder = SigLIPEmbedder()
    repository = QdrantVectorRepository()
    extractor = DecordFrameExtractor(fps_sample_rate=1.0)
    app.state.process = ProcessVideoService(extractor, embedder, repository)
    app.state.search = SearchVideoService(embedder, repository)
    yield


app = FastAPI(title="Video Search Engine", lifespan=lifespan)
app.include_router(router)


@app.get("/")
async def read_root():
    return {"message": "Video Search Engine API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
