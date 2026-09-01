"""grounded answering with a local vlm or llm"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

from copilot.retrieval.base import Evidence


# the gap between a grounded and a fabricated answer is wide here
MIN_FAITHFULNESS = 0.5


@dataclass
class Answer:
    text: str
    # evidence whose page the answer actually cited, in retrieval order
    evidence_used: list[Evidence]
    insufficient_evidence: bool = False
    # pages cited that were not in the evidence, meaning an invented source
    unsupported_pages: list[int] = field(default_factory=list)

    # share of the answer's content words found in the cited evidence
    faithfulness: float = 1.0
    # content words in the answer found nowhere in what it cited
    unsupported_terms: list[str] = field(default_factory=list)

    # log of the tool calls that produced this answer, empty for the fixed pipeline
    tool_calls: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """whether this answer is actually backed by the evidence"""
        if self.insufficient_evidence:
            return True
        if not self.evidence_used:
            return False
        return self.faithfulness >= MIN_FAITHFULNESS


class AnswerGenerator(ABC):
    """generating a grounded answer to a question from evidence only"""

    @abstractmethod
    def generate(self, question: str, evidence: list[Evidence]) -> Answer: ...

    def generate_stream(
        self, question: str, evidence: list[Evidence]
    ) -> Iterator[tuple[str, str] | tuple[str, Answer]]:
        """like generate, yielding tokens then one done event, not every generator supports it"""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")
