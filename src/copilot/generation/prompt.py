"""turning retrieved evidence into a grounded prompt"""

from dataclasses import dataclass

from copilot.retrieval.base import Evidence, EvidenceKind

# a sentinel so refusal is detected exactly, not by pattern-matching prose
INSUFFICIENT_MARKER = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = f"""You are a technical documentation assistant for industrial equipment manuals.

You answer ONLY from the EVIDENCE provided in the user message. The evidence is
the complete set of facts available to you.

Rules:
1. Use only the EVIDENCE. Do not use prior knowledge about pumps, motors, or
   any equipment. If the evidence does not contain the answer, do not supply
   one from memory.
2. If the evidence is insufficient to answer, reply with exactly
   {INSUFFICIENT_MARKER} and nothing else. This is the correct answer when the
   evidence is off-topic or only partially relevant. Do not guess.
3. Cite the page for every claim, in square brackets, like [page 37]. Only cite
   page numbers that appear in the EVIDENCE.
4. Separate fact from inference. State what the manual says directly, and when
   you draw a conclusion the manual does not state outright, mark it clearly
   (for example "This suggests ...").
5. Be concise and technical. Do not pad the answer.

Worked example of a grounded answer:
EVIDENCE:
[1] (page 12) The intake filter must be replaced every 500 operating hours.
QUESTION: How often is the intake filter replaced?
ANSWER: The intake filter must be replaced every 500 operating hours [page 12].

Worked example when the evidence does not answer the question:
EVIDENCE:
[1] (page 12) The intake filter must be replaced every 500 operating hours.
QUESTION: What is the warranty period?
ANSWER: {INSUFFICIENT_MARKER}

Every sentence you write must end with a [page N] citation, or be
{INSUFFICIENT_MARKER}.
"""

EVIDENCE_HEADER = "EVIDENCE:"
QUESTION_HEADER = "QUESTION:"


@dataclass
class RenderedPrompt:
    system: str
    user: str
    # images to attach when the generator is a vlm, ignored otherwise
    image_paths: list[str]


def _render_text_evidence(index: int, evidence: Evidence) -> str:
    body = " ".join((evidence.text or "").split())
    return f"[{index}] (page {evidence.page_number}) {body}"


def _render_image_evidence(index: int, evidence: Evidence) -> str:
    """announcing an image with its page even when no caption exists"""
    caption = " ".join((evidence.text or "").split())
    described = f"Diagram: {caption}" if caption else "Diagram (no caption available)"
    return f"[{index}] (page {evidence.page_number}) {described}"


FACTS_HEADER = "COMPUTED FACTS (from tools; trusted, no page citation needed):"


def build_prompt(
    question: str,
    evidence: list[Evidence],
    computed_facts: list[str] | None = None,
) -> RenderedPrompt:
    """computed_facts are trusted tool results, rendered separately from citable evidence"""
    lines: list[str] = []
    image_paths: list[str] = []

    for index, item in enumerate(evidence, start=1):
        if item.kind is EvidenceKind.IMAGE:
            lines.append(_render_image_evidence(index, item))
            if item.image_path:
                image_paths.append(item.image_path)
        else:
            lines.append(_render_text_evidence(index, item))

    body = "\n".join(lines) if lines else "(no evidence retrieved)"
    user = f"{EVIDENCE_HEADER}\n{body}"

    if computed_facts:
        facts_body = "\n".join(f"- {fact}" for fact in computed_facts)
        user += f"\n\n{FACTS_HEADER}\n{facts_body}"

    user += f"\n\n{QUESTION_HEADER} {question.strip()}"
    return RenderedPrompt(system=SYSTEM_PROMPT, user=user, image_paths=image_paths)


def cited_pages(answer: str) -> set[int]:
    """page numbers the answer claims to cite, in [page n] form"""
    import re

    return {int(match) for match in re.findall(r"\[page\s+(\d+)\]", answer, flags=re.IGNORECASE)}
