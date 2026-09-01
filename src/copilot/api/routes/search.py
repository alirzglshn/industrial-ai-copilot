"""semantic search over ingested manuals, retrieval only"""

import logging

from fastapi import APIRouter, Depends

from copilot.api.deps import require_retrieval_stack
from copilot.retrieval.base import Evidence
from copilot.schemas.search import EvidenceOut, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


def _to_out(evidence: Evidence) -> EvidenceOut:
    return EvidenceOut(
        kind=evidence.kind.value,
        document_id=evidence.document_id,
        page_number=evidence.page_number,
        score=evidence.score,
        fused_score=evidence.fused_score,
        chunk_id=evidence.chunk_id,
        text=evidence.text,
        image_id=evidence.image_id,
        image_path=evidence.image_path,
    )


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, stack=Depends(require_retrieval_stack)) -> SearchResponse:
    results = stack.multimodal.retrieve(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        include_images=request.include_images,
    )
    return SearchResponse(query=request.query, results=[_to_out(e) for e in results])
