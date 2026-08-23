"""A deterministic embedder for tests.

Retrieval plumbing (filtering, top-k, ordering, re-index cleanup) is worth
testing on every run, but loading a real model is slow and downloads weights.
This produces genuine cosine similarity from lexical overlap, so ranking
assertions are meaningful without a model. Real semantic behaviour is covered
separately by the integration tests, which use the actual model.
"""

import math
import re
from zlib import crc32

from copilot.retrieval.embedder import TextEmbedder

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
            # crc32 rather than hash(): str hashing is salted per process, and
            # a fixture whose vectors change between runs is not a fixture.
            vector[crc32(word.encode()) % self._dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            # Qdrant rejects zero-length vectors under cosine distance.
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embedded_queries.append(text)
        # The real embedder prepends the BGE instruction prefix here. This one
        # deliberately does not: those prefix tokens would dominate a
        # bag-of-words vector and flatten the ranking these tests assert on.
        # The prefix contract is covered directly in test_embedder.py.
        return self._vector(text)
