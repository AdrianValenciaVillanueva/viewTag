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

## Pendiente

- Dockerfile multi-stage + GitHub Actions CI (lint, type-check, tests)
- Logging estructurado (JSON) + métricas Prometheus `/metrics`
- Rate limiting / API keys (`slowapi`)
- Graceful shutdown en lifespan (cleanup Qdrant, modelo)
- Pre-commit hooks (ruff, mypy, tests)
- Cache de embeddings (Redis / disco) para reprocesamiento
- Tests de carga (locust/k6)
- Async real en infraestructura (extractor, embedder, repo)
