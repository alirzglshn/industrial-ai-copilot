"""the shape of a ground-truth question, and loading a set of them"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalQuestion:
    id: str
    question: str
    # retrieval: normal question with known gold pages
    # refusal: off-topic, the system should decline rather than guess
    # calculation: needs the calculator, checked by substring not pages
    category: str

    # gold pages as document filename stem, page number, resolved after ingestion
    expected_pages: list[tuple[str, int]] = field(default_factory=list)

    # for refusal, insufficient_evidence must be set; for calculation, a substring match
    expect_insufficient: bool = False
    answer_must_contain: list[str] = field(default_factory=list)


def load_questions(path: Path) -> list[EvalQuestion]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = []
    for item in raw:
        questions.append(
            EvalQuestion(
                id=item["id"],
                question=item["question"],
                category=item["category"],
                expected_pages=[tuple(pair) for pair in item.get("expected_pages", [])],
                expect_insufficient=item.get("expect_insufficient", False),
                answer_must_contain=item.get("answer_must_contain", []),
            )
        )
    return questions
