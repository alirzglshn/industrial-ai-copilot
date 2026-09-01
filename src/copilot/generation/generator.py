"""generating a grounded answer with a local text model or vlm"""

import logging
from collections.abc import Iterator
from functools import lru_cache

from copilot.core.config import get_settings
from copilot.generation.base import MIN_FAITHFULNESS, Answer, AnswerGenerator
from copilot.generation.faithfulness import measure
from copilot.generation.grounding import ground
from copilot.generation.local_lm import get_local_lm
from copilot.generation.prompt import INSUFFICIENT_MARKER, build_prompt
from copilot.retrieval.base import Evidence

logger = logging.getLogger(__name__)


def _is_refusal(text: str) -> bool:
    """whether the model declined for lack of evidence, matched by substring"""
    return INSUFFICIENT_MARKER in text.upper()


def _finish(question: str, raw: str, evidence: list[Evidence]) -> Answer:
    text = raw.strip()

    if not text:
        # empty generation reported as insufficient, not a confident blank
        logger.warning("Model returned an empty answer for %r", question)
        return Answer(text="", evidence_used=[], insufficient_evidence=True)

    if _is_refusal(text):
        return Answer(text=text, evidence_used=[], insufficient_evidence=True)

    result = ground(text, evidence)
    if result.has_unsupported_citations:
        logger.warning(
            "Answer cited pages not present in the evidence: %s", result.unsupported_pages
        )

    # a resolved citation proves the page was retrieved, not that it supports the claim
    support = measure(text, result.evidence_used)
    if result.evidence_used and support.score < MIN_FAITHFULNESS:
        logger.warning(
            "Answer cites real pages but is not supported by them (%.2f); "
            "terms absent from the cited evidence: %s",
            support.score,
            support.unsupported_terms[:10],
        )

    return Answer(
        text=text,
        evidence_used=result.evidence_used,
        insufficient_evidence=False,
        unsupported_pages=result.unsupported_pages,
        faithfulness=support.score,
        unsupported_terms=support.unsupported_terms,
    )


def stream_answer(
    lm,
    question: str,
    evidence: list[Evidence],
    max_new_tokens: int,
    computed_facts: list[str] | None = None,
) -> Iterator[tuple[str, str] | tuple[str, Answer]]:
    """token-by-token generation, ending in the same checked answer chat() produces"""
    if not evidence and not computed_facts:
        yield ("done", _no_evidence_answer())
        return

    prompt = build_prompt(question, evidence, computed_facts=computed_facts)
    pieces: list[str] = []
    for piece in lm.chat_stream(prompt.system, prompt.user, max_new_tokens):
        pieces.append(piece)
        yield ("token", piece)

    yield ("done", _finish(question, "".join(pieces), evidence))


def _no_evidence_answer() -> Answer:
    """refusing when retrieval found nothing, rather than inviting a guess"""
    return Answer(
        text=INSUFFICIENT_MARKER,
        evidence_used=[],
        insufficient_evidence=True,
    )


class LocalLlmAnswerGenerator(AnswerGenerator):
    def __init__(self, model_name: str, max_new_tokens: int = 300) -> None:
        # cached by model name, shared with the agent when configured the same
        self.lm = get_local_lm(model_name)
        self.max_new_tokens = max_new_tokens

    def generate(self, question: str, evidence: list[Evidence]) -> Answer:
        if not evidence:
            return _no_evidence_answer()

        prompt = build_prompt(question, evidence)
        raw = self.lm.chat(prompt.system, prompt.user, self.max_new_tokens)
        return _finish(question, raw, evidence)

    def generate_stream(self, question: str, evidence: list[Evidence]):
        yield from stream_answer(self.lm, question, evidence, self.max_new_tokens)


class VlmAnswerGenerator(AnswerGenerator):
    """answering with the retrieved images actually in context"""

    def __init__(self, model_name: str, max_new_tokens: int = 300, max_images: int = 3) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        # each image costs hundreds of tokens of context and seconds of cpu
        self.max_images = max_images

    def _load_images(self, paths: list[str]):
        from PIL import Image as PILImage

        images = []
        for path in paths[: self.max_images]:
            try:
                with PILImage.open(path) as opened:
                    images.append(opened.convert("RGB"))
            except Exception:
                logger.warning("Could not load image %s for answering", path)
        return images

    def generate(self, question: str, evidence: list[Evidence]) -> Answer:
        if not evidence:
            return _no_evidence_answer()

        prompt = build_prompt(question, evidence)
        images = self._load_images(prompt.image_paths)

        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt.user})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt.system}]},
            {"role": "user", "content": content},
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=text, images=images or None, return_tensors="pt"
        )

        with self._torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )

        completion = generated[0][inputs["input_ids"].shape[-1] :]
        raw = self.processor.decode(completion, skip_special_tokens=True)
        return _finish(question, raw, evidence)


@lru_cache(maxsize=1)
def get_answer_generator() -> AnswerGenerator:
    # no settings parameter, a settings instance is not hashable for lru_cache
    settings = get_settings()
    if settings.use_vlm_for_answers:
        return VlmAnswerGenerator(
            settings.vlm_model,
            max_new_tokens=settings.answer_max_new_tokens,
            max_images=settings.answer_max_images,
        )
    return LocalLlmAnswerGenerator(
        settings.answer_model, max_new_tokens=settings.answer_max_new_tokens
    )
