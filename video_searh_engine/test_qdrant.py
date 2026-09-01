
from src.infrastructure.database.qdrant_repo import QdrantVectorRepository  

def main():
    # Ejemplo de uso
    repo = QdrantVectorRepository(storage_path="./qdrant_data", collection_name="video_frames", vector_size=768)
    repo._ensure_collection(vector_size=768)


if __name__ == "__main__":
    main()