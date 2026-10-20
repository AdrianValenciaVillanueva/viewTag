import logging
import sys
from typing import Any
import structlog
from pythonjsonlogger import jsonlogger

def configure_logging(json_logs: bool = True, log_level: str = "INFO") -> None:
    """
    Configura structlog para logging JSON estructurado.

    Args:
        json_logs: Si True, output JSON; si False, output consola legible (dev).
        log_level: Nivel mínimo de log (DEBUG, INFO, WARNING, ERROR).
    """
    #configuracion basica
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )

#procesadores compartidos
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        #configuracion para logs en JSON
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        #configuracion para logs en consola legible
        shared_processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Obtiene un logger estructurado para el módulo dado.

    Usage:
        logger = get_logger(__name__)
        logger.info("processing_video", video_name="demo.mp4", frames=120)
    """
    return structlog.get_logger(name)
