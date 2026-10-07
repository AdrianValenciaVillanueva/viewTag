from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from application.process_video import ProcessVideoService
from application.search_video import SearchVideoService

#models
class ProcessRequest(BaseModel):
    video_path: str = Field(min_length = 1)
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


router = APIRouter()


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


@router.get("/search", response_model=list[SearchHit])
def search_video(
    q: str = Query(min_length=1),
    limit: int = Query(5, ge=1, le=50),
    service: SearchVideoService = Depends(get_search_service),
):
    try:
        return service.execute(q, limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

