# src/api/rate_limit.py
"""Rate limiting configuration using slowapi."""
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from config.settings import settings


def get_client_id(request: Request) -> str:
    """
    Identificador único para rate limiting.
    Combina IP + API Key si existe, sino solo IP.
    """
    # Intentar obtener API Key del header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"apikey:{api_key}"
    # Fallback a IP
    return get_remote_address(request)


# Limiter global - usa client_id (IP + API Key)
limiter = Limiter(
    key_func=get_client_id,
    default_limits=[f"{settings.rate_limit_requests_per_minute}/minute"],
    storage_uri=settings.rate_limit_storage_uri,
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Handler personalizado para 429 Too Many Requests."""
    # RateLimitExceeded no tiene retry_after directamente, calcularlo
    retry_after = 60  # default 60 segundos para límite por minuto
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": f"Rate limit exceeded: {exc.detail}",
            "retry_after": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )


def setup_rate_limiting(app: FastAPI) -> None:
    """Configura rate limiting en la app FastAPI."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)