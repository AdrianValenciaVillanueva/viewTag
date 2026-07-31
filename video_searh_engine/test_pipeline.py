import time
from typing import List

from src.infrastructure.ia.siglip_embedder import SigLIPEmbedder
from src.infrastructure.video.decord_extractor import DecordFrameExtractor



def dot_product(vec1:List[float], vec2: List[float]) -> float:
    """
    calcula el producto escalar entre 2 vectores
    SigLIPEmbedder entrega los vectores normalizador
    el producto escalar es la similitud de coseno
    """
    return sum(a * b for a, b in zip(vec1, vec2))



def main():

    VIDEO_PATH = "video_searh_engine/test.mp4"

    SEARCH_QUERY = "code"

    #inicializar componentes
    print("cargando componentes\n")
    extractor = DecordFrameExtractor(fps_sample_rate = 1)
    embedder = SigLIPEmbedder()
    print("componentes listos\n")

    print("extraccion de fotogramas\n")
    start_time = time.time()

    try:

        frames = extractor.extract(video_path = VIDEO_PATH)

    except Exception as e:
        print(f"error al abrir el video {e}")
        return

    extraction_time = time.time() - start_time

    print(f" extraccion completa: {len(frames)} frames en: {extraction_time:.2f} segundos")

    #generar el embedding para cada fotograma
    print("generando vectores de imagenes con SigLIP")

    start_time = time.time()

    frame_embedding = []

    for frame in frames:
        vector = embedder.embed_image(frame.image_bytes)
        frame_embedding.append((frame, vector))

    ia_time = time.time() - start_time
    print(f"vectore generados en: {ia_time:.2f} segundos")

    print("convertir la busqueda a vector")
    query_vector = embedder.embed_text(SEARCH_QUERY)

    #calcular similud
    results = []
    for frame, img_vector in frame_embedding:
        score = dot_product(img_vector, query_vector)
        results.append((frame, score))

    results.sort(key=lambda x: x[1], reverse=True)

    print("mostrando los mejores 3 resultados")

    for rank, (frame, score) in enumerate(results[:3], start=1):
        print(f" top {rank} segundo: {frame.timestamp_seconds} score {score:.4f}")




if __name__ == "__main__":
    main()







