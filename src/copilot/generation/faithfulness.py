"""does the cited evidence actually support the answer, by word overlap"""

import re
from dataclasses import dataclass

from copilot.retrieval.base import Evidence

_WORD = re.compile(r"[a-z0-9]+")

# citation markers are structure, not claims to be scored
_CITATION = re.compile(r"\[page\s+\d+\]", flags=re.IGNORECASE)

# function words carry no evidential weight
_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has
    have how if in into is it its may must not of on or should so such than
    that the their then there these they this to use used using was were what
    when where which while who will with would you your
    """.split()
)


def _stem(word: str) -> str:
    """stripping common inflections so a paraphrase is not a fabrication"""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _content_words(text: str) -> set[str]:
    """words as they appear, lowercased, not stemmed, for legible reporting"""
    return {
        word
        for word in _WORD.findall(text.lower())
        # short tokens are mostly units and markers, not meaningful matches
        if len(word) > 2 and word not in _STOPWORDS
    }


@dataclass
class FaithfulnessResult:
    # share of the answer's content words found in the cited evidence
    score: float
    # content words found nowhere in what was cited
    unsupported_terms: list[str]


def measure(answer: str, evidence_used: list[Evidence]) -> FaithfulnessResult:
    """overlap between an answer and the evidence it cited"""
    answer_words = _content_words(_CITATION.sub("", answer))
    if not answer_words:
        # nothing claimed, nothing unsupported
        return FaithfulnessResult(score=1.0, unsupported_terms=[])

    evidence_words: set[str] = set()
    for item in evidence_used:
        if item.text:
            evidence_words |= _content_words(item.text)

    if not evidence_words:
        # no readable text in the cited evidence to support a claim
        return FaithfulnessResult(score=0.0, unsupported_terms=sorted(answer_words))

    evidence_stems = {_stem(word) for word in evidence_words}
    missing = {word for word in answer_words if _stem(word) not in evidence_stems}
    supported = len(answer_words) - len(missing)
    return FaithfulnessResult(
        score=supported / len(answer_words),
        unsupported_terms=sorted(missing),
    )
