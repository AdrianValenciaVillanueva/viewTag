import io
import torch 
from  typing import List
from PIL import Image
from transformers import AutoProcessor, SiglipModel
from transformers.image_utils import load_image

from src.domain.interfaces.embedder import VectorEmbedderInterface

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

    #check to batch processing
    def embed_image(self, image:Image.Image) -> List[float]:
        """convierte los bytes de una imagen en una lista de numeros embed"""
        image = image

        #procesar la imagen 
        #pasamos la imagen a tensor con torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            #check
            image_features = self.model.get_image_features(**inputs)
            print(image_features.shape)

            #check
            # Si devuelve un contenedor en lugar del Tensor directo:
            if not isinstance(image_features, torch.Tensor):
                image_features = getattr(image_features, "pooler_output", getattr(image_features, "image_embeds", image_features[0]))
            #normalizamos el vector
            image_features = image_features/image_features.norm(dim=1, keepdim=True)

            #devolvemos el vector
            return image_features.squeeze().cpu().tolist()
        
    def embed_text(self, text:str) -> List[float]:
        """convierte un texto en una lista de numeros embed"""
        #procesar el texto
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.inference_mode():
            #check
            text_features = self.model.get_text_features(**inputs)
            # Si devuelve un contenedor en lugar del Tensor directo:
            #check
            if not isinstance(text_features, torch.Tensor):
                text_features = getattr(text_features, "pooler_output", getattr(text_features, "text_embeds", text_features[0]))
            #normalizamos el vector
            text_features = text_features/text_features.norm(dim=1, keepdim=True)

            #devolvemos el vector
            return text_features.squeeze(0).cpu().tolist()