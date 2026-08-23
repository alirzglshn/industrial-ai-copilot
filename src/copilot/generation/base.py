"""Phase 5: grounded answering with a local VLM/LLM.

The generator must answer only from the passed-in evidence, say so
explicitly when the evidence is insufficient, and cite the page(s) it used.
Implemented in Phase 5. Defined now so copilot.agent can be written against
a stable contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from copilot.retrieval.base import Evidence


@dataclass
class Answer:
    text: str
    evidence_used: list[Evidence]
    insufficient_evidence: bool = False


class AnswerGenerator(ABC):
    """Generates a grounded answer to a question from retrieved evidence only."""

    @abstractmethod
    def generate(self, question: str, evidence: list[Evidence]) -> Answer: ...
