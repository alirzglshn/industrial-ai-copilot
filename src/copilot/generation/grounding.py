"""checking an answer's citations against the evidence it was given"""

from dataclasses import dataclass, field

from copilot.generation.prompt import cited_pages
from copilot.retrieval.base import Evidence


@dataclass
class GroundingResult:
    # evidence whose page the answer actually cited
    evidence_used: list[Evidence] = field(default_factory=list)
    # pages cited that were never in the evidence, an invented source
    unsupported_pages: list[int] = field(default_factory=list)

    @property
    def has_unsupported_citations(self) -> bool:
        return bool(self.unsupported_pages)


def ground(answer: str, evidence: list[Evidence]) -> GroundingResult:
    """resolving an answer's citations against the evidence it was shown"""
    claimed = cited_pages(answer)
    if not claimed:
        return GroundingResult()

    available = {item.page_number for item in evidence}
    supported = claimed & available
    unsupported = sorted(claimed - available)

    # keeping retrieval order, not page order, so the most relevant stays first
    used = [item for item in evidence if item.page_number in supported]

    return GroundingResult(evidence_used=used, unsupported_pages=unsupported)
