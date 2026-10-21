# Configuration settings using pydantic-settings
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Environment variables take precedence over defaults.
    Use a .env file for local development (auto-loaded by pydantic-settings).
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Qdrant configuration
    qdrant_storage_path: str = "./qdrant_data"
    qdrant_collection_name: str = "video_frames"
    qdrant_vector_size: int = 768  # SigLIP base patch16-224 output dimension
    
    # Model configuration
    siglip_model_name: str = "google/siglip-base-patch16-224"
    siglip_model_path: str = ""  # Ruta local al modelo (opcional, para uso offline)
    
    # Video processing configuration
    fps_sample_rate: float = 1.0  # frames per second to extract
    frame_batch_size: int = 64    # frames per batch during extraction
    image_batch_size: int = 32    # images per batch during embedding
    upsert_batch_size: int = 256  # points per batch during upsert
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    
    # Search limits
    search_default_limit: int = 5
    search_max_limit: int = 50
    search_min_limit: int = 1
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_storage_uri: str = "memory://"
    
    # Text processing
    text_max_length: int = 64


# Singleton instance
settings = Settings()