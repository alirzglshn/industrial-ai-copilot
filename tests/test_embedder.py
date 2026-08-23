"""Contract tests for the sentence-transformers embedder.

The model itself is stubbed: what matters here is the asymmetry BGE requires
(queries prefixed, passages not) and that vectors are normalized. Whether the
real model retrieves well is measured by the integration tests.
"""

import numpy as np
import pytest

from copilot.retrieval.embedder import BGE_QUERY_PREFIX, SentenceTransformerEmbedder


class StubModel:
    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.encoded: list[list[str]] = []
        self.normalize_flags: list[bool] = []

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, batch_size=32, normalize_embeddings=False, show_progress_bar=False):
        self.encoded.append(list(texts))
        self.normalize_flags.append(normalize_embeddings)
        return np.ones((len(texts), self.dimension), dtype=float)


@pytest.fixture
def embedder() -> SentenceTransformerEmbedder:
    instance = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
    instance.model = StubModel()
    instance.query_prefix = BGE_QUERY_PREFIX
    instance.batch_size = 32
    return instance


def test_dimension_comes_from_the_model(embedder: SentenceTransformerEmbedder) -> None:
    assert embedder.dimension == 8


def test_queries_are_prefixed(embedder: SentenceTransformerEmbedder) -> None:
    embedder.embed_query("what causes overheating")

    assert embedder.model.encoded[-1] == [
        f"{BGE_QUERY_PREFIX}what causes overheating"
    ]


def test_passages_are_not_prefixed(embedder: SentenceTransformerEmbedder) -> None:
    embedder.embed_documents(["overheating is caused by low airflow"])

    assert embedder.model.encoded[-1] == ["overheating is caused by low airflow"]


def test_vectors_are_normalized(embedder: SentenceTransformerEmbedder) -> None:
    embedder.embed_documents(["a"])
    embedder.embed_query("b")

    # Cosine distance in Qdrant assumes this.
    assert all(embedder.model.normalize_flags)


def test_embedding_nothing_skips_the_model(embedder: SentenceTransformerEmbedder) -> None:
    assert embedder.embed_documents([]) == []
    assert embedder.model.encoded == []
