from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from api.rate_limit import limiter, validate_api_key
from application.process_video import ProcessVideoService
from application.search_video import SearchVideoService
from config.settings import settings


# Models
class ProcessRequest(BaseModel):
    video_path: str = Field(min_length=1)
    video_name: str | None = None


class ProcessResponse(BaseModel):
    video_name: str
    frames_processed: int
    embeddings_saved: int


class BatchProcessItem(BaseModel):
    video_path: str
    status: str  # "ok" | "error"
    result: ProcessResponse | None = None
    error: str | None = None


class BatchProcessRequest(BaseModel):
    videos: list[ProcessRequest] = Field(
        min_length=1, max_length=settings.max_batch_size, description="Lista de videos a procesar"
    )


class BatchProcessResponse(BaseModel):
    results: list[BatchProcessItem]
    total: int
    succeeded: int
    failed: int


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
@limiter.limit("100/minute")
def health_check(request: Request):
    return {"status": "ok"}


def get_process_service(request: Request) -> ProcessVideoService:
    return request.app.state.process


def get_search_service(request: Request) -> SearchVideoService:
    return request.app.state.search


@router.post(
    "/process-video", response_model=ProcessResponse, dependencies=[Depends(validate_api_key)]
)
@limiter.limit("10/minute")
async def process_video(
    request: Request,
    body: ProcessRequest,
    service: ProcessVideoService = Depends(get_process_service),
):
    try:
        return await service.execute(body.video_path, body.video_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post(
    "/process-videos", response_model=BatchProcessResponse, dependencies=[Depends(validate_api_key)]
)
@limiter.limit("5/minute")
async def process_videos(
    request: Request,
    body: BatchProcessRequest,
    service: ProcessVideoService = Depends(get_process_service),
):
    results: list[BatchProcessItem] = []
    succeeded = 0
    failed = 0
    for video in body.videos:
        try:
            result = await service.execute(video.video_path, video.video_name)
            results.append(
                BatchProcessItem(
                    video_path=video.video_path, status="ok", result=ProcessResponse(**result)
                )
            )
            succeeded += 1
        except Exception as e:
            results.append(
                BatchProcessItem(video_path=video.video_path, status="error", error=str(e))
            )
            failed += 1

    return BatchProcessResponse(
        results=results, total=len(body.videos), succeeded=succeeded, failed=failed
    )


@router.get("/search", response_model=SearchPageResponse, dependencies=[Depends(validate_api_key)])
@limiter.limit("30/minute")
async def search_video(
    request: Request,
    q: str = Query(min_length=1, description="Texto a buscar"),
    limit: int = Query(
        default=settings.search_default_limit,
        ge=settings.search_min_limit,
        le=settings.search_max_limit,
        description="Máximo número de resultados",
    ),
    offset: int = Query(default=0, ge=0, description="Número de resultados a saltar (paginación)"),
    video_name: str | None = Query(
        default=None, description="Filtrar resultados por nombre de video"
    ),
    service: SearchVideoService = Depends(get_search_service),
):
    try:
        page = await service.execute(q, limit=limit, offset=offset, video_name=video_name)
        return SearchPageResponse(
            results=[
                SearchHit(
                    frame_id=r.frame_id,
                    video_name=r.video_name,
                    timestamp_seconds=r.timestamp_seconds,
                    score=r.score,
                )
                for r in page.results
            ],
            total=page.total,
            limit=page.limit,
            offset=page.offset,
            has_more=page.has_more,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
