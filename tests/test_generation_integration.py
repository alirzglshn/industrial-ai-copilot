"""generation with a real local model

deselected by default, downloads weights on first run:
    pytest -m integration

asserts mechanics, not answer quality: chat template, completion-only
decoding, bounded and reproducible generation, and answer shape
"""

import os
from pathlib import Path

import pytest

from copilot.generation.generator import LocalLlmAnswerGenerator
from copilot.generation.prompt import INSUFFICIENT_MARKER, QUESTION_HEADER
from tests.test_prompt import image_evidence, text_evidence

pytestmark = pytest.mark.integration

HUB_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# preferring a local copy under models/ so this runs offline, smaller model as fallback
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
LOCAL_CANDIDATES = ("Qwen2.5-1.5B-Instruct", "Qwen2.5-0.5B-Instruct")


def _model_source() -> str:
    for name in LOCAL_CANDIDATES:
        if (MODELS_DIR / name / "model.safetensors").exists():
            return str(MODELS_DIR / name)
    return HUB_ID


MODEL = os.environ.get("ANSWER_MODEL") or _model_source()

OVERHEATING = (
    "Overheating is most commonly caused by insufficient cooling airflow "
    "across the motor fins. A blocked intake filter reduces airflow by up to "
    "sixty percent and will trigger the thermal cutout."
)


@pytest.fixture(scope="module")
def generator() -> LocalLlmAnswerGenerator:
    # module-scoped, since loading the model dominates the runtime of these tests
    return LocalLlmAnswerGenerator(MODEL, max_new_tokens=120)


def test_produces_a_non_empty_answer(generator: LocalLlmAnswerGenerator) -> None:
    answer = generator.generate(
        "What causes the pump to overheat?", [text_evidence(37, OVERHEATING)]
    )

    assert answer.text.strip()


def test_only_the_completion_is_returned(generator: LocalLlmAnswerGenerator) -> None:
    """the prompt must not be echoed back into the answer"""
    answer = generator.generate(
        "What causes the pump to overheat?", [text_evidence(37, OVERHEATING)]
    )

    assert QUESTION_HEADER not in answer.text
    assert "EVIDENCE:" not in answer.text


def test_generation_is_bounded(generator: LocalLlmAnswerGenerator) -> None:
    answer = generator.generate("Explain everything.", [text_evidence(37, OVERHEATING)])

    # 120 new tokens cannot decode to anything near this many characters
    assert len(answer.text) < 4000


def test_greedy_decoding_is_reproducible(generator: LocalLlmAnswerGenerator) -> None:
    """sampling is off, so the same evidence must give the same answer"""
    evidence = [text_evidence(37, OVERHEATING)]

    first = generator.generate("What causes overheating?", evidence)
    second = generator.generate("What causes overheating?", evidence)

    assert first.text == second.text


def test_no_evidence_refuses_without_calling_the_model(
    generator: LocalLlmAnswerGenerator,
) -> None:
    answer = generator.generate("What causes overheating?", [])

    assert answer.insufficient_evidence
    assert INSUFFICIENT_MARKER in answer.text


def test_answer_carries_only_resolvable_citations(
    generator: LocalLlmAnswerGenerator,
) -> None:
    """whatever the model writes, every returned citation is real"""
    evidence = [text_evidence(37, OVERHEATING)]

    answer = generator.generate("What causes overheating?", evidence)

    available = {item.page_number for item in evidence}
    assert all(item.page_number in available for item in answer.evidence_used)
    assert all(page not in available for page in answer.unsupported_pages)


def test_image_evidence_does_not_break_the_text_model(
    generator: LocalLlmAnswerGenerator,
) -> None:
    """a diagram reaches the text model as a page reference, not pixels"""
    answer = generator.generate(
        "Which page shows the diagram?",
        [text_evidence(37, OVERHEATING), image_evidence(38, "impeller clearance")],
    )

    assert answer.text.strip()
