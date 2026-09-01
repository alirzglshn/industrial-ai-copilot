"""one retriever over text and images, merged by reciprocal rank fusion

fusing by rank rather than raw score, since bge and clip similarities come
from different models and are not on comparable scales
"""

import logging
from dataclasses import replace

from copilot.retrieval.base import Evidence, EvidenceKind, Retriever
from copilot.retrieval.image_retriever import ImageRetriever, PageImageSource
from copilot.retrieval.retriever import VectorRetriever

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60


def _key(evidence: Evidence) -> tuple:
    if evidence.kind is EvidenceKind.IMAGE:
        return ("image", evidence.image_id)
    return ("text", evidence.chunk_id)


def reciprocal_rank_fusion(
    rankings: list[list[Evidence]], k: int = DEFAULT_RRF_K
) -> list[Evidence]:
    """merging several ranked lists into one, scoring by rank rather than similarity"""
    fused: dict[tuple, float] = {}
    best: dict[tuple, Evidence] = {}

    for ranking in rankings:
        for rank, evidence in enumerate(ranking, start=1):
            identity = _key(evidence)
            fused[identity] = fused.get(identity, 0.0) + 1.0 / (k + rank)
            # keeping the representative with a real similarity score over a page-context stand-in
            existing = best.get(identity)
            if existing is None or evidence.score > existing.score:
                best[identity] = evidence

    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [replace(best[identity], fused_score=score) for identity, score in ordered]


class MultimodalRetriever(Retriever):
    def __init__(
        self,
        text_retriever: VectorRetriever,
        image_retriever: ImageRetriever | None = None,
        page_images: PageImageSource | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        image_top_k: int = 5,
        page_context_pages: int = 3,
    ) -> None:
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever
        self.page_images = page_images
        self.rrf_k = rrf_k
        self.image_top_k = image_top_k
        self.page_context_pages = page_context_pages

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
        include_images: bool = True,
    ) -> list[Evidence]:
        if not query.strip():
            return []

        text_hits = self.text_retriever.retrieve(query, top_k=top_k, document_id=document_id)
        if not include_images:
            return text_hits

        rankings: list[list[Evidence]] = [text_hits]

        if self.image_retriever is not None:
            try:
                rankings.append(
                    self.image_retriever.retrieve(
                        query, top_k=self.image_top_k, document_id=document_id
                    )
                )
            except Exception:
                # image search is an enhancement, its failure must not cost the text results
                logger.warning("Image search failed; returning text evidence only", exc_info=True)

        if self.page_images is not None and text_hits:
            pages = [(hit.document_id, hit.page_number) for hit in text_hits]
            try:
                rankings.append(
                    self.page_images.for_pages(pages[: self.page_context_pages])
                )
            except Exception:
                logger.warning("Page-context image lookup failed", exc_info=True)

        return reciprocal_rank_fusion(rankings, k=self.rrf_k)[:top_k]
