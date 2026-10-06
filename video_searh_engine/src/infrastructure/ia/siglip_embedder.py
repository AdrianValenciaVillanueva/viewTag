import torch
from typing import Any, Dict, List
from PIL import Image
from transformers import AutoProcessor, SiglipModel


from domain.interfaces.embedder import VectorEmbedderInterface

_IMAGE_BATCH_SIZE = 32
_TEXT_MAX_LENGTH = 64


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
    Estrategia para crear vectores con SigLIP mediante hugging face
    """
    def __init__(self, model_name:str = "google/siglip-base-patch16-224"):
        #usar gpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #cargar modelador de imagenes y un modelo preentrenado
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SiglipModel.from_pretrained(model_name).to(self.device).eval()

    def embed_image(self, images: List[Image.Image]) -> List[List[float]]:
        """convierte una lista de imagenes en una lista de vectores embed"""
        if not images:
            return []

        all_vectors: List[List[float]] = []
        with torch.inference_mode():
            for i in range(0, len(images), _IMAGE_BATCH_SIZE):
                chunk = images[i : i + _IMAGE_BATCH_SIZE]
                inputs = _move_to_device(
                    self.processor(images=chunk, return_tensors="pt"), self.device
                )

                image_features = _output_to_tensor(
                    self.model.get_image_features(**inputs),
                    ["pooler_output", "image_embeds"],
                )

                #normalizamos el vector
                norms = image_features.norm(dim=1, keepdim=True).clamp(min=1e-12)
                image_features = image_features / norms
                #devolvemos el vector
                all_vectors.extend(image_features.cpu().tolist())

        return all_vectors

    def embed_text(self, text:str) -> List[float]:
        """convierte el texto de busqueda en un numero en el mismo espacio vectorial"""
        if not text:
            raise ValueError("text no puede estar vacío")

        #procesar el texto
        inputs = _move_to_device(
            self.processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=_TEXT_MAX_LENGTH,
            ),
            self.device,
        )

        with torch.inference_mode():
            text_features = _output_to_tensor(
                self.model.get_text_features(**inputs),
                ["pooler_output", "text_embeds"],
            )
            #normalizamos el vector
            norms = text_features.norm(dim=1, keepdim=True).clamp(min=1e-12)
            text_features = text_features / norms

            #devolvemos el vector
            return text_features.squeeze(0).cpu().tolist()
