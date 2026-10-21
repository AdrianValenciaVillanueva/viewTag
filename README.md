# viewTag — Buscador semántico de videos

Sube un video, y después busca momentos dentro de él escribiendo texto.
Por ejemplo: procesas una grabación de tu pantalla y buscas `"código"`.
La API te devuelve en qué segundo aparece y con qué puntuación.

## ¿Cómo funciona?

1. **Extrae fotogramas** del video (1 por segundo) con Decord.
2. **Convierte** cada fotograma y tu texto de búsqueda en vectores
   numéricos (embeddings) con SigLIP, un modelo que entiende imágenes
   y texto en el mismo "idioma" matemático.
3. **Compara** el vector de tu búsqueda contra los vectores de los
   fotogramas guardados en Qdrant (base de datos vectorial) y te
   devuelve los más parecidos, ordenados por puntuación.

```
video.mp4
  │  POST /process-video
  ▼
[frames] → [vectores SigLIP] → [Qdrant]
                                     │
  GET /search?q=codigo ─────────────┘
  ▼
[{ frame, segundo, score }, ...]
```

## Requisitos

- Python >= 3.11
- Los pesos del modelo se descargan solos de Hugging Face
  (`google/siglip-base-patch16-224`) la primera vez que arrancas.

## Configuración

Toda la configuración se centraliza en `src/config/settings.py` usando **pydantic-settings**.
Los valores por defecto funcionan *out-of-the-box*; se pueden sobrescribir con variables de entorno
o un archivo `.env` en `video_searh_engine/`.

```bash
# Ejemplo: copiar plantilla y editar
cp video_searh_engine/.env.example video_searh_engine/.env
```

Variables principales (ver `.env.example` para la lista completa):

| Variable | Default | Descripción |
|----------|---------|-------------|
| `QDRANT_STORAGE_PATH` | `./qdrant_data` | Directorio donde Qdrant persiste datos |
| `SIGLIP_MODEL_NAME` | `google/siglip-base-patch16-224` | Modelo HF para embeddings |
| `FPS_SAMPLE_RATE` | `1.0` | Frames a extraer por segundo |
| `IMAGE_BATCH_SIZE` | `32` | Lote para embeddings de imágenes |
| `UPSERT_BATCH_SIZE` | `256` | Lote para upsert en Qdrant |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | Host/puerto de la API |
| `SEARCH_DEFAULT_LIMIT` | `5` | Límite por defecto en `/search` |

## Instalación

### Local (Python)

```bash
cd video_searh_engine

# crear y activar entorno (Windows)
python -m venv ..\..\.venv  # o usa el .venv existente en la raíz
..\..\.venv\Scripts\activate

pip install -e .
pip install -r tests/requirements.txt

# opcional: configurar variables de entorno
cp .env.example .env
# editar .env si se desea cambiar rutas, modelo, lotes, etc.
```

### Con Docker (recomendado para producción)

```bash
cd video_searh_engine

# Build imagen CPU
docker build -t viewtag .

# O con GPU (requiere nvidia-container-toolkit)
docker build --build-arg ENABLE_GPU=1 -t viewtag:gpu .

# Run con docker-compose (persistencia + hot reload en dev)
docker compose up --build

# O run manual
docker run -p 8000:8000 \
  -v $(pwd)/qdrant_data:/app/qdrant_data \
  -v $(pwd)/.env:/app/.env:ro \
  viewtag
```

## Uso

### Opción 1: CLI (recomendado)
Instala el paquete en modo editable y usa el comando global:

```bash
cd video_searh_engine
pip install -e .
viewtag-api
# Usa config del .env (host, port, reload)
# docs interactivas en http://localhost:8001/docs (o tu API_PORT)
```

### Opción 2: Python directo
```bash
cd video_searh_engine
python src/api/main.py
```

### Opción 3: Uvicorn manual
```bash
cd video_searh_engine
uvicorn api.main:app --reload
# docs interactivas en http://localhost:8000/docs
```

> Todas las opciones usan la config centralizada (`.env` / `settings`).
> La opción 1 (`viewtag-api`) es la más cómoda: no requiere `PYTHONPATH` ni rutas relativas.

Procesa un video:

```bash
curl -X POST localhost:8000/process-video \
  -H "Content-Type: application/json" \
  -d '{"video_path": "mi/video.mp4"}'
# {"video_name": "video", "frames_processed": 120, "embeddings_saved": 120}
```

Busca por texto:

```bash
curl "localhost:8000/search?q=codigo&limit=3"
# [{"frame_id": "frame_90", "video_name": "video",
#   "timestamp_seconds": 3.0, "score": 0.91}, ...]
```

## Endpoints

| Método | Ruta             | Qué hace                                        |
|--------|------------------|-------------------------------------------------|
| POST   | `/process-video` | Extrae frames, los vectoriza y los guarda      |
| GET    | `/search`        | Busca frames parecidos a un texto (paginado, con filtro) |
| GET    | `/health`        | Health check para load balancers / k8s         |
| GET    | `/`              | Salud del servicio                              |

### `GET /search` — Parámetros

| Parámetro    | Tipo    | Requerido | Default | Descripción                                    |
|--------------|---------|-----------|---------|------------------------------------------------|
| `q`          | string  | ✅        | —       | Texto a buscar (min 1 char)                    |
| `limit`      | integer | ❌        | 5       | Resultados por página (1–50)                   |
| `offset`     | integer | ❌        | 0       | Resultados a saltar (paginación)               |
| `video_name` | string  | ❌        | —       | Filtrar solo frames de este video              |

### `GET /search` — Respuesta

```json
{
  "results": [
    { "frame_id": "frame_90", "video_name": "video", "timestamp_seconds": 3.0, "score": 0.91 }
  ],
  "total": 120,
  "limit": 5,
  "offset": 0,
  "has_more": true
}
```

Errores: `422` datos mal formados, `400` parámetro inválido, `404` video no encontrado, `500` error interno.

## Estructura

```
video_searh_engine/src/
├── config/                  # configuración centralizada (pydantic-settings)
│   ├── __init__.py
│   └── settings.py          # Settings class + singleton `settings`
├── domain/                  # reglas del negocio (no depende de nada externo)
│   ├── models.py            # Frame (bytes JPEG + decode lazy) y SearchResult
│   └── interfaces/          # contratos: extractor, embedder, repository
├── application/             # casos de uso (solo conocen domain/)
│   ├── process_video.py     # extraer → vectorizar → guardar
│   └── search_video.py      # vectorizar texto → buscar similares
├── infrastructure/          # implementaciones concretas
│   ├── video/decord_extractor.py  # frames JPEG, validado y por lotes
│   ├── ia/siglip_embedder.py      # embeddings normalizados, lotes de 32
│   └── database/qdrant_repo.py    # Qdrant local, upsert por lotes
├── observability/           # logging, métricas, tracing
│   ├── __init__.py
│   └── logging.py           # structlog config (JSON/console), get_logger()
└── api/                     # FastAPI: DTOs, rutas y composition root
    ├── routes.py
    └── main.py              # crea los singletons una vez (lifespan)
```

La regla de oro: `application` nunca importa `infrastructure`.
Solo `main.py` conoce las clases concretas y las inyecta.
Así los casos de uso se prueban con fakes, sin GPU ni disco.

## Decisiones técnicas

- **Frames como JPEG, no bytes crudos**: `Frame.image_bytes` guarda JPEG
  decodificable (q95) en vez del buffer crudo de píxeles, que no conserva
  tamaño ni formato y rompía la reconstrucción.
- **Todo por lotes**: extracción (64), embeddings (32) y guardado (256)
  para no agotar RAM/VRAM con videos largos.
- **Re-procesar no duplica**: cada punto lleva
  `id = uuid5(video_name + frame_id)`, así que procesar dos veces
  el mismo video sobrescribe en lugar de duplicar.
- **Validar y fallar rápido**: rutas, fps, longitudes y límites se
  validan en cada capa (`422/400/404`) en vez de fallar a medias.

## Tests

```bash
cd video_searh_engine
python -m pytest tests -v
```

**39 tests** (32 unitarios + 7 integración):

| Suite | Tests | Qué cubre |
|-------|-------|-----------|
| `test_fixtures.py` | 12 | Fixtures base (frames, vectores, mocks) |
| `test_process_video.py` | 8 | `ProcessVideoService`: flujo, validaciones, errores |
| `test_search_video.py` | 12 | `SearchVideoService`: paginación, filtro video_name, SearchPage |
| `integration/test_pipeline.py` | 7 | Pipeline real: video sintético + Qdrant en memoria |

Los scripts `test_pipeline.py` y `test_qdrant.py` en la raíz son pruebas manuales adicionales, no parte de la suite.

## Docker

### Multi-stage Dockerfile

```dockerfile
# Build: docker build -t viewtag .
# GPU:   docker build --build-arg ENABLE_GPU=1 -t viewtag:gpu .
# Run:   docker run -p 8000:8000 -v $(pwd)/qdrant_data:/app/qdrant_data viewtag
```

**Características:**
- **Multi-stage**: base → builder → gpu-builder → runtime (imagen final ~500MB CPU, ~3GB GPU)
- **Non-root user**: `appuser` para seguridad
- **Healthcheck**: `curl /health` cada 30s
- **Entry point**: `viewtag-api` (usa config del `.env`)
- **GPU opcional**: `--build-arg ENABLE_GPU=1` instala PyTorch CUDA 12.1

### Docker Compose (desarrollo)

```bash
docker compose up --build    # Primera vez / cambios en Dockerfile
docker compose up -d         # Detached
docker compose logs -f       # Ver logs
docker compose down -v       # Para y borra volumes
```

Incluye persistencia de Qdrant (`qdrant_data` volume), hot reload de código (`./src:/app/src:ro`), y montaje de `.env`.

## CI/CD — GitHub Actions

Pipeline completo en `.github/workflows/ci.yml`:

| Job | Qué hace | Trigger |
|-----|----------|---------|
| `lint` | `ruff check` + `ruff format --check` | Push/PR a `main` |
| `type-check` | `mypy --strict src/` | Push/PR a `main` |
| `unit-tests` | `pytest tests/unit -v` | Push/PR a `main` |
| `integration-tests` | `pytest tests/integration -v` (modelo real + Qdrant mem) | Push/PR a `main` |
| `docker-build` | Build imagen CPU + smoke test `curl /health` | Push/PR a `main` |
| `docker-build-gpu` | Build + test imagen GPU | Manual / Schedule |
| `release` | Build/push a **GHCR** + GitHub Release | Tag `v*` |

**Release automático:** Al hacer push de un tag `v1.0.0`, se publican:
- `ghcr.io/user/viewtag:v1.0.0` y `:latest` (CPU)
- `ghcr.io/user/viewtag:v1.0.0-gpu` y `:gpu-latest` (GPU)

## Pre-commit Hooks

Calidad automática antes de cada commit:

```bash
# Instalar (una vez)
pip install pre-commit
pre-commit install

# Probar manualmente
pre-commit run --all-files
```

**Hooks configurados:**
- `ruff` + `ruff-format` — Lint y formato (reemplaza flake8/black/isort)
- `mypy --strict` — Type checking en `src/`
- `check-merge-conflict` — Detecta `<<<<<<< HEAD` sin resolver
- `gitleaks` — Escanea secrets (API keys, tokens)
- `check-yaml/toml`, `end-of-file-fixer`, `trailing-whitespace`, `mixed-line-ending(lf)`

## Script Helper — `./dev.sh`

```bash
./dev.sh <comando>

Comandos:
  setup         → venv + deps + pre-commit install
  test          → Todos los tests (unit + integration)
  test-unit     → Solo unit tests
  test-int      → Solo integration tests
  lint          → ruff check
  format        → ruff format + fix
  type-check    → mypy --strict
  check         → lint + type-check + test-unit (todo)
  run           → viewtag-api (usa .env)
  run-docker    → docker compose up --build
  build-docker  → docker build -t viewtag .
  clean         → Limpia caches (__pycache__, .pytest_cache, etc.)
  pre-commit    → Instala/actualiza hooks
  help          → Muestra esta ayuda
```

Ejemplo flujo diario:
```bash
./dev.sh setup       # Primera vez
./dev.sh check       # Antes de commit/PR
./dev.sh run         # Desarrollo local
./dev.sh run-docker  # Test en contenedor
./dev.sh build-docker  # docker build -t viewtag .
```

## Pendiente

### Producción / Robustez
- (completado) Rate limiting / API keys (`slowapi`)

### Escalabilidad / Performance
- Cache de embeddings (Redis / disco) para reprocesamiento
- Async real en infraestructura (extractor, embedder, repo)
- Batch processing endpoint (`POST /process-videos`)

### Calidad de vida
- Tests de carga (locust/k6)
- OpenAPI examples + tags en Swagger UI
- Webhook callbacks (`POST /callback` al terminar procesar)
- Multi-tenancy real (`tenant_id` + aislamiento datos)

## Observability

### Logging estructurado (JSON)

Logging con `structlog` + `python-json-logger`. Output JSON por línea:

```json
{"timestamp": "2026-10-20T18:53:12.205739Z", "level": "info", "logger": "api.main", "event": "startup", "embedder_loaded": true, "qdrant_connected": true}
{"timestamp": "2026-10-20T18:53:45.123456Z", "level": "info", "logger": "api.main", "event": "shutdown_started"}
{"timestamp": "2026-10-20T18:53:45.124567Z", "level": "info", "logger": "api.main", "event": "qdrant_client_closed"}
{"timestamp": "2026-10-20T18:53:45.125678Z", "level": "info", "logger": "api.main", "event": "shutdown_complete"}
```

- **Producción**: JSON lines (compatible con ELK, Datadog, Loki, CloudWatch)
- **Desarrollo**: Consola colorizada legible
- Configurable via `configure_logging(json_logs=True/False, log_level="INFO")`
- Uso: `logger = get_logger(__name__); logger.info("evento", key=value, ...)`

### Métricas Prometheus (`/metrics`)

Endpoint expuesto automáticamente con `prometheus-fastapi-instrumentator`:

```bash
curl http://localhost:8001/metrics
```

Métricas incluidas:

| Métrica | Tipo | Descripción |
|---------|------|-------------|
| `http_requests_total` | Counter | Requests por handler, método, status |
| `http_request_duration_seconds` | Histogram | Latencia por handler, método |
| `http_requests_inprogress` | Gauge | Requests concurrentes |
| `http_request_size_bytes` | Histogram | Tamaño request body |
| `http_response_size_bytes` | Histogram | Tamaño response body |
| `http_request_duration_highr_seconds` | Histogram | Latencia alta resolución |

Excluye `/health` y `/metrics` para evitar ruido.

### Graceful Shutdown

Cierre ordenado en `lifespan` de FastAPI:

```json
{"event": "shutdown_started", "logger": "api.main", "level": "info", "timestamp": "..."}
{"event": "qdrant_client_closed", "logger": "api.main", "level": "info", "timestamp": "..."}
{"event": "shutdown_complete", "logger": "api.main", "level": "info", "timestamp": "..."}
```

- Cierra cliente Qdrant liberando lock en `./qdrant_data`
- Logs estructurados para auditoría
- Compatible con `docker stop` (SIGTERM), `kubectl delete pod`, `Ctrl+C`

## Script Helper — `./dev.sh`

```bash
./dev.sh <comando>

Comandos:
  setup         → venv + deps + pre-commit install
  test          → Todos los tests (unit + integration)
  test-unit     → Solo unit tests
  test-int      → Solo integration tests
  lint          → ruff check
  format        → ruff format + fix
  type-check    → mypy --strict
  check         → lint + type-check + test-unit (todo)
  run           → viewtag-api (usa .env)
  run-docker    → docker compose up --build
  build-docker  → docker build -t viewtag .
  clean         → Limpia caches (__pycache__, .pytest_cache, etc.)
  pre-commit    → Instala/actualiza hooks
  help          → Muestra esta ayuda
```

Ejemplo flujo diario:
```bash
./dev.sh setup       # Primera vez
./dev.sh check       # Antes de commit/PR
./dev.sh run         # Desarrollo local
./dev.sh run-docker  # Test en contenedor
./dev.sh build-docker  # docker build -t viewtag .
```

## Pendiente

### Producción / Robustez
- (completado) Rate limiting / API keys (`slowapi`)

### Escalabilidad / Performance
- Cache de embeddings (Redis / disco) para reprocesamiento
- Async real en infraestructura (extractor, embedder, repo)
- Batch processing endpoint (`POST /process-videos`)

### Calidad de vida
- Tests de carga (locust/k6)
- OpenAPI examples + tags en Swagger UI
- Webhook callbacks (`POST /callback` al terminar procesar)
- Multi-tenancy real (`tenant_id` + aislamiento datos)

## Rate Limiting + API Keys

Rate limiting con `slowapi` para proteger la API de abuso y permitir multi-tenancy.

### Configuración

```bash
# .env
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_STORAGE_URI=memory://  # o redis://localhost:6379 en producción
```

### Endpoints con rate limit

| Endpoint | Límite | Descripción |
|----------|--------|-------------|
| `/health` | 100/min | Health check frecuente |
| `/search` | 30/min | Búsqueda semántica |
| `/process-video` | 10/min | Procesamiento costoso |

### API Keys (header `X-API-Key`)

Permite buckets de rate limit separados por cliente:

```bash
# Sin API Key → bucket por IP (límite compartido)
curl http://localhost:8001/health

# Con API Key → bucket propio (límite independiente)
curl -H "X-API-Key: mi-api-key-secreta" http://localhost:8001/search?q=test
```

- Cada API Key tiene su propio bucket de rate limit independiente
- Permite multi-tenancy: cada cliente tiene su propio límite
- Header: `X-API-Key: tu-api-key-secreta`

### Response 429 (Rate Limited)

```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded: 100 per 1 minute",
  "retry_after": 60
}
```

Headers:
```
Retry-After: 60
```

### Configuración avanzada

```bash
# Storage en Redis (producción)
RATE_LIMIT_STORAGE_URI=redis://localhost:6379

# Límite personalizado por endpoint (en código)
@limiter.limit("5/minute")
@router.get("/endpoint-costoso")
async def endpoint_costoso(...):
```

## Script Helper — `./dev.sh`