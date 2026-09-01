"""deterministic embedders and stand-ins for tests, real semantics covered by integration tests"""

import math
import re
from pathlib import Path
from zlib import crc32

from copilot.generation.base import Answer, AnswerGenerator
from copilot.generation.generator import _finish, _no_evidence_answer
from copilot.retrieval.base import Evidence
from copilot.retrieval.captioner import ImageCaptioner
from copilot.retrieval.embedder import TextEmbedder
from copilot.retrieval.image_embedder import ImageEmbedder

_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbedder(TextEmbedder):
    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension
        self.embedded_queries: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for word in _WORD.findall(text.lower()):
            # crc32 rather than hash(), since str hashing is salted per process
            vector[crc32(word.encode()) % self._dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            # qdrant rejects zero-length vectors under cosine distance
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embedded_queries.append(text)
        # skipping the real bge prefix, which would dominate this bag-of-words vector
        return self._vector(text)


def _bag_of_words_vector(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    for word in _WORD.findall(text.lower()):
        vector[crc32(word.encode()) % dimension] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


class HashingImageEmbedder(ImageEmbedder):
    """standing in for clip's shared text and image space, via bag-of-words descriptions"""

    def __init__(self, dimension: int = 64, descriptions: dict[str, str] | None = None) -> None:
        self._dimension = dimension
        self.descriptions = descriptions or {}
        self.embedded_queries: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def describe(self, path: str, text: str) -> None:
        self.descriptions[str(Path(path))] = text

    def _description_for(self, path: str) -> str:
        return self.descriptions.get(str(Path(path)), Path(path).stem)

    def embed_images(self, paths: list[str]) -> list[list[float] | None]:
        results: list[list[float] | None] = []
        for path in paths:
            if not Path(path).exists():
                results.append(None)
                continue
            results.append(_bag_of_words_vector(self._description_for(path), self._dimension))
        return results

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_bag_of_words_vector(text, self._dimension) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embedded_queries.append(text)
        return self.embed_texts([text])[0]


class ScriptedAnswerGenerator(AnswerGenerator):
    """returning a canned generation, then running it through the real grounding path"""

    def __init__(self, output: str = "Low airflow causes it [page 1].") -> None:
        self.output = output
        self.calls: list[tuple[str, list[Evidence]]] = []

    def generate(self, question: str, evidence: list[Evidence]) -> Answer:
        self.calls.append((question, list(evidence)))
        if not evidence:
            return _no_evidence_answer()
        return _finish(question, self.output, evidence)

    def generate_stream(self, question: str, evidence: list[Evidence]):
        self.calls.append((question, list(evidence)))
        if not evidence:
            yield ("done", _no_evidence_answer())
            return
        # one token chunk is enough for the event sequence, real granularity needs a real model
        yield ("token", self.output)
        yield ("done", _finish(question, self.output, evidence))


class ScriptedLocalLM:
    """standing in for localcausallm: scripted completions instead of a loaded model"""

    def __init__(self, outputs: list[str] | None = None) -> None:
        # each call to chat() consumes the next scripted output, the last one repeats after that
        self.outputs = outputs or ["[]"]
        self.calls: list[tuple[str, str, int]] = []

    def chat(self, system: str, user: str, max_new_tokens: int) -> str:
        index = min(len(self.calls), len(self.outputs) - 1)
        self.calls.append((system, user, max_new_tokens))
        return self.outputs[index]

    def chat_stream(self, system: str, user: str, max_new_tokens: int):
        # one chunk is enough for the streaming event sequence, real granularity needs a real model
        yield self.chat(system, user, max_new_tokens)


class FallbackOnlyPlanner:
    """a planner stand-in returning llmplanner's default fallback plan, without calling a model"""

    def __init__(self, include_images: bool = True) -> None:
        self.include_images = include_images

    def plan(self, question: str):
        from copilot.agent.planner import default_fallback_plan

        return default_fallback_plan(question, include_images=self.include_images)


class StubCaptioner(ImageCaptioner):
    """returning a fixed caption per path, or none to model a captioning failure"""

    def __init__(self, captions: dict[str, str | None] | None = None) -> None:
        self.captions = captions or {}
        self.captioned: list[str] = []

    def caption(self, paths: list[str]) -> list[str | None]:
        self.captioned.extend(paths)
        return [self.captions.get(str(Path(path)), "a diagram of a pump") for path in paths]
