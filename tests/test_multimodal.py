from copilot.retrieval.base import Evidence, EvidenceKind
from copilot.retrieval.image_retriever import PageImageSource
from copilot.retrieval.multimodal import (
    DEFAULT_RRF_K,
    MultimodalRetriever,
    reciprocal_rank_fusion,
)


def text(chunk_id: str, page: int = 1, score: float = 0.7) -> Evidence:
    return Evidence(
        kind=EvidenceKind.TEXT,
        document_id="doc-a",
        page_number=page,
        score=score,
        chunk_id=chunk_id,
        text=f"chunk {chunk_id}",
    )


def image(image_id: str, page: int = 1, score: float = 0.3) -> Evidence:
    return Evidence(
        kind=EvidenceKind.IMAGE,
        document_id="doc-a",
        page_number=page,
        score=score,
        image_id=image_id,
        image_path=f"/images/{image_id}.png",
    )


class StubTextRetriever:
    def __init__(self, results: list[Evidence]) -> None:
        self.results = results
        self.calls: list[tuple] = []

    def retrieve(self, query, top_k=5, document_id=None):
        self.calls.append((query, top_k, document_id))
        return self.results[:top_k]


class StubImageRetriever(StubTextRetriever):
    pass


class StubPageImages(PageImageSource):
    def __init__(self, results: list[Evidence], explode: bool = False) -> None:
        self.results = results
        self.explode = explode
        self.requested: list = []

    def for_pages(self, pages):
        self.requested.append(list(pages))
        if self.explode:
            raise RuntimeError("lookup failed")
        return self.results


# --- fusion ---------------------------------------------------------------


def test_fusion_ranks_by_rank_not_by_score() -> None:
    """A high raw score from one model must not outrank a top hit from another.

    The image here scores 0.3 against the text's 0.7, but both are their own
    ranking's first result, so they tie on rank.
    """
    fused = reciprocal_rank_fusion([[text("c1", score=0.7)], [image("i1", score=0.3)]])

    assert {e.fused_score for e in fused} == {1.0 / (DEFAULT_RRF_K + 1)}


def test_appearing_in_two_rankings_outranks_appearing_in_one() -> None:
    both = image("i1")
    fused = reciprocal_rank_fusion(
        [[text("c1"), text("c2")], [image("i2"), both], [both]]
    )

    assert fused[0].image_id == "i1"
    assert fused[0].fused_score > fused[1].fused_score


def test_duplicates_are_merged_not_repeated() -> None:
    fused = reciprocal_rank_fusion([[image("i1")], [image("i1")], [image("i1")]])

    assert len(fused) == 1
    assert fused[0].fused_score == 3 * (1.0 / (DEFAULT_RRF_K + 1))


def test_merged_item_keeps_the_more_informative_score() -> None:
    """A page-context image carries score 0.0; CLIP's score should survive."""
    fused = reciprocal_rank_fusion([[image("i1", score=0.0)], [image("i1", score=0.42)]])

    assert fused[0].score == 0.42


def test_native_scores_are_preserved_alongside_the_fused_one() -> None:
    fused = reciprocal_rank_fusion([[text("c1", score=0.71)]])

    assert fused[0].score == 0.71
    assert fused[0].fused_score is not None
    assert fused[0].fused_score != fused[0].score


def test_results_are_sorted_by_fused_score() -> None:
    fused = reciprocal_rank_fusion([[text("c1"), text("c2"), text("c3")]])

    scores = [e.fused_score for e in fused]
    assert scores == sorted(scores, reverse=True)
    assert [e.chunk_id for e in fused] == ["c1", "c2", "c3"]


def test_fusing_nothing_returns_nothing() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# --- retriever ------------------------------------------------------------


def test_combines_text_image_and_page_context() -> None:
    retriever = MultimodalRetriever(
        text_retriever=StubTextRetriever([text("c1")]),
        image_retriever=StubImageRetriever([image("i1")]),
        page_images=StubPageImages([image("i2", page=1)]),
    )

    results = retriever.retrieve("cooling", top_k=10)

    assert {e.chunk_id for e in results if e.kind is EvidenceKind.TEXT} == {"c1"}
    assert {e.image_id for e in results if e.kind is EvidenceKind.IMAGE} == {"i1", "i2"}


def test_page_context_is_asked_for_the_pages_text_matched() -> None:
    page_images = StubPageImages([])
    retriever = MultimodalRetriever(
        text_retriever=StubTextRetriever([text("c1", page=37), text("c2", page=4)]),
        image_retriever=None,
        page_images=page_images,
    )

    retriever.retrieve("cooling", top_k=5)

    assert page_images.requested == [[("doc-a", 37), ("doc-a", 4)]]


def test_include_images_false_skips_both_image_paths() -> None:
    image_retriever = StubImageRetriever([image("i1")])
    page_images = StubPageImages([image("i2")])
    retriever = MultimodalRetriever(
        text_retriever=StubTextRetriever([text("c1")]),
        image_retriever=image_retriever,
        page_images=page_images,
    )

    results = retriever.retrieve("cooling", top_k=5, include_images=False)

    assert all(e.kind is EvidenceKind.TEXT for e in results)
    assert image_retriever.calls == []
    assert page_images.requested == []
    # Text-only means nothing was fused, so no fused score is invented.
    assert all(e.fused_score is None for e in results)


def test_text_survives_a_failing_image_search() -> None:
    class Broken:
        def retrieve(self, *args, **kwargs):
            raise RuntimeError("CLIP exploded")

    retriever = MultimodalRetriever(
        text_retriever=StubTextRetriever([text("c1")]),
        image_retriever=Broken(),
        page_images=None,
    )

    results = retriever.retrieve("cooling", top_k=5)

    assert [e.chunk_id for e in results] == ["c1"]


def test_text_survives_a_failing_page_context_lookup() -> None:
    retriever = MultimodalRetriever(
        text_retriever=StubTextRetriever([text("c1")]),
        image_retriever=None,
        page_images=StubPageImages([], explode=True),
    )

    assert [e.chunk_id for e in retriever.retrieve("cooling", top_k=5)] == ["c1"]


def test_works_with_no_image_side_at_all() -> None:
    retriever = MultimodalRetriever(
        text_retriever=StubTextRetriever([text("c1")]), image_retriever=None, page_images=None
    )

    assert [e.chunk_id for e in retriever.retrieve("cooling", top_k=5)] == ["c1"]


def test_blank_query_returns_nothing() -> None:
    image_retriever = StubImageRetriever([image("i1")])
    retriever = MultimodalRetriever(
        text_retriever=StubTextRetriever([text("c1")]), image_retriever=image_retriever
    )

    assert retriever.retrieve("   ") == []
    assert image_retriever.calls == []


def test_result_count_is_capped_at_top_k() -> None:
    retriever = MultimodalRetriever(
        text_retriever=StubTextRetriever([text(f"c{i}") for i in range(10)]),
        image_retriever=StubImageRetriever([image(f"i{i}") for i in range(10)]),
        page_images=StubPageImages([image(f"p{i}") for i in range(10)]),
        image_top_k=10,
    )

    assert len(retriever.retrieve("cooling", top_k=3)) == 3


def test_page_context_is_skipped_when_text_found_nothing() -> None:
    page_images = StubPageImages([image("i2")])
    retriever = MultimodalRetriever(
        text_retriever=StubTextRetriever([]), image_retriever=None, page_images=page_images
    )

    assert retriever.retrieve("cooling", top_k=5) == []
    assert page_images.requested == []


def test_document_filter_reaches_both_retrievers() -> None:
    text_retriever = StubTextRetriever([text("c1")])
    image_retriever = StubImageRetriever([image("i1")])
    retriever = MultimodalRetriever(
        text_retriever=text_retriever, image_retriever=image_retriever, image_top_k=4
    )

    retriever.retrieve("cooling", top_k=2, document_id="doc-a")

    assert text_retriever.calls == [("cooling", 2, "doc-a")]
    assert image_retriever.calls == [("cooling", 4, "doc-a")]
