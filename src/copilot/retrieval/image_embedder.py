"""embedding images and queries into clip's shared space

clip is weak on schematics and line drawings, out of distribution from its
natural-photo training set, so the multimodal retriever also surfaces images
by page context rather than relying on clip scoring alone
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

# clip's text encoder is fixed at 77 tokens, truncated rather than left to raise
CLIP_MAX_TOKENS = 77


class ImageEmbedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """vector width, needed to size the image collection"""

    @abstractmethod
    def embed_images(self, paths: list[str]) -> list[list[float] | None]:
        """embedding image files, none in place of any file that could not be read"""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """embedding text into the same space, for text to image search and captions"""

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class ClipImageEmbedder(ImageEmbedder):
    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        # imported lazily so this module can be imported without torch present
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.batch_size = batch_size

    @property
    def dimension(self) -> int:
        return int(self.model.config.projection_dim)

    @staticmethod
    def _projected(features):
        """the projected embedding, whichever shape this transformers version returns"""
        pooler_output = getattr(features, "pooler_output", None)
        return features if pooler_output is None else pooler_output

    def _normalize(self, features) -> list[list[float]]:
        features = self._projected(features)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return [row.tolist() for row in features]

    def embed_images(self, paths: list[str]) -> list[list[float] | None]:
        from PIL import Image as PILImage

        if not paths:
            return []

        results: list[list[float] | None] = [None] * len(paths)
        loaded: list[tuple[int, object]] = []
        for position, path in enumerate(paths):
            try:
                with PILImage.open(Path(path)) as image:
                    loaded.append((position, image.convert("RGB").copy()))
            except Exception:
                # a missing or corrupt file must not abort the whole batch
                continue

        for start in range(0, len(loaded), self.batch_size):
            batch = loaded[start : start + self.batch_size]
            inputs = self.processor(images=[image for _, image in batch], return_tensors="pt")
            with self._torch.no_grad():
                features = self.model.get_image_features(**inputs)
            for (position, _), vector in zip(batch, self._normalize(features)):
                results[position] = vector

        return results

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=CLIP_MAX_TOKENS,
            )
            with self._torch.no_grad():
                features = self.model.get_text_features(**inputs)
            vectors.extend(self._normalize(features))
        return vectors


@lru_cache(maxsize=1)
def get_image_embedder(model_name: str) -> ImageEmbedder:
    """cached on the model name, not a settings instance"""
    return ClipImageEmbedder(model_name)
