import torch
from typing import Any, Dict, List
from PIL import Image
from transformers import AutoProcessor, SiglipModel

from domain.interfaces.embedder import VectorEmbedderInterface
from config.settings import settings


def _move_to_device(inputs: Any, device: str) -> Any:
    """Mueve los tensores del processor al device sin depender de BatchEncoding.to()."""
    to_fn = getattr(inputs, "to", None)
    if callable(to_fn):
        try:
            return to_fn(device)
        except Exception:
            pass
    if isinstance(inputs, Dict):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    data = getattr(inputs, "data", None)
    if isinstance(data, dict):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
    return inputs


def _output_to_tensor(output: Any, attrs: List[str]) -> torch.Tensor:
    """Extrae el Tensor de la salida del modelo sin evaluar índices por defecto."""
    if isinstance(output, torch.Tensor):
        return output
    for attr in attrs:
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Salida inesperada del modelo: {type(output)}")


class SigLIPEmbedder(VectorEmbedderInterface):
    """
    Estrategia para crear vectores con SigLIP mediante Hugging Face.
    Configuración centralizada vía settings; parámetros opcionales para tests.
    """

    def __init__(
        self,
        model_name: str | None = None,
        image_batch_size: int | None = None,
        text_max_length: int | None = None,
        model_path: str | None = None,
    ):
        self.model_name = model_name or settings.siglip_model_name
        self.image_batch_size = image_batch_size or settings.image_batch_size
        self.text_max_length = text_max_length or settings.text_max_length
        # Usar model_path pasado por parámetro, o el de settings si no está vacío
        self.model_path = model_path or (settings.siglip_model_path if settings.siglip_model_path else None)
        
        # Usar GPU si está disponible
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Cargar processor y modelo preentrenado
        if self.model_path:
            self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
            self.model = SiglipModel.from_pretrained(self.model_path, local_files_only=True).to(self.device).eval()
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = SiglipModel.from_pretrained(self.model_name).to(self.device).eval()

    def embed_image(self, images: List[Image.Image]) -> List[List[float]]:
        """Convierte una lista de imágenes en una lista de vectores embedding normalizados."""
        if not images:
            return []

        all_vectors: List[List[float]] = []
        with torch.inference_mode():
            for i in range(0, len(images), self.image_batch_size):
                chunk = images[i : i + self.image_batch_size]
                inputs = _move_to_device(
                    self.processor(images=chunk, return_tensors="pt"), self.device
                )

                image_features = _output_to_tensor(
                    self.model.get_image_features(**inputs),
                    ["pooler_output", "image_embeds"],
                )

                # Normalización L2 para similitud de coseno
                norms = image_features.norm(dim=1, keepdim=True).clamp(min=1e-12)
                image_features = image_features / norms
                all_vectors.extend(image_features.cpu().tolist())

        return all_vectors

    def embed_text(self, text: str) -> List[float]:
        """Convierte el texto de búsqueda en un vector en el mismo espacio vectorial."""
        if not text:
            raise ValueError("text no puede estar vacío")

        inputs = _move_to_device(
            self.processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.text_max_length,
            ),
            self.device,
        )

        with torch.inference_mode():
            text_features = _output_to_tensor(
                self.model.get_text_features(**inputs),
                ["pooler_output", "text_embeds"],
            )
            norms = text_features.norm(dim=1, keepdim=True).clamp(min=1e-12)
            text_features = text_features / norms

            return text_features.squeeze(0).cpu().tolist()
