from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from application.process_video import ProcessVideoService
from application.search_video import SearchVideoService
from config.settings import settings
from domain.interfaces.repository import SearchPage

# Models
class ProcessRequest(BaseModel):
    video_path: str = Field(min_length=1)
    video_name: str | None = None


class ProcessResponse(BaseModel):
    video_name: str
    frames_processed: int
    embeddings_saved: int


class SearchHit(BaseModel):
    frame_id: str
    video_name: str
    timestamp_seconds: float
    score: float


class SearchPageResponse(BaseModel):
    """Paginated search response with metadata."""
    results: list[SearchHit]
    total: int
    limit: int
    offset: int
    has_more: bool


router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


def get_process_service(request: Request) -> ProcessVideoService:
    return request.app.state.process


def get_search_service(request: Request) -> SearchVideoService:
    return request.app.state.search


@router.post("/process-video", response_model=ProcessResponse)
def process_video(
    body: ProcessRequest,
    service: ProcessVideoService = Depends(get_process_service),
):
    try:
        return service.execute(body.video_path, body.video_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/search", response_model=SearchPageResponse)
def search_video(
    q: str = Query(min_length=1, description="Texto a buscar"),
    limit: int = Query(
        default=settings.search_default_limit,
        ge=settings.search_min_limit,
        le=settings.search_max_limit,
        description="Máximo número de resultados",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Número de resultados a saltar (paginación)",
    ),
    video_name: str | None = Query(
        default=None,
        description="Filtrar resultados por nombre de video",
    ),
    service: SearchVideoService = Depends(get_search_service),
):
    try:
        page = service.execute(q, limit=limit, offset=offset, video_name=video_name)
        return SearchPageResponse(
            results=[
                SearchHit(
                    frame_id=r.frame_id,
                    video_name=r.video_name,
                    timestamp_seconds=r.timestamp_seconds,
                    score=r.score,
                )for r in page.results
            ],
            total=page.total,
            limit=page.limit,
            offset=page.offset,
            has_more=page.has_more,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

