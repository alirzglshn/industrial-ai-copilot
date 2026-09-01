"""optional vision-language captions for extracted images, off by default"""

import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageCaptioner(ABC):
    @abstractmethod
    def caption(self, paths: list[str]) -> list[str | None]:
        """describing each image, none where the file could not be captioned"""


class TransformersImageCaptioner(ImageCaptioner):
    """captioning with a vision-language model loaded directly, not via pipeline()"""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 40,
        prompt: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt

    def _caption_one(self, path: str) -> str | None:
        from PIL import Image as PILImage

        with PILImage.open(Path(path)) as opened:
            image = opened.convert("RGB")

        if self.prompt:
            inputs = self.processor(images=image, text=self.prompt, return_tensors="pt")
        else:
            inputs = self.processor(images=image, return_tensors="pt")

        with self._torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        text = self.processor.decode(generated[0], skip_special_tokens=True).strip()
        if self.prompt and text.startswith(self.prompt):
            # stripping the echoed instruction some models prepend to the caption
            text = text[len(self.prompt) :].strip()
        return text or None

    def caption(self, paths: list[str]) -> list[str | None]:
        captions: list[str | None] = []
        for path in paths:
            try:
                captions.append(self._caption_one(path))
            except Exception:
                # one unreadable image must not stop the document from indexing
                logger.warning("Could not caption image %s", path, exc_info=True)
                captions.append(None)
        return captions


@lru_cache(maxsize=1)
def get_image_captioner(
    model_name: str, max_new_tokens: int = 40, prompt: str | None = None
) -> ImageCaptioner:
    """cached on plain hashable arguments, not a settings instance"""
    return TransformersImageCaptioner(model_name, max_new_tokens=max_new_tokens, prompt=prompt)
