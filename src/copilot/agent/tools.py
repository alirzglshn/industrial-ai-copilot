"""the five tools an agent can call"""

import ast
import logging
import operator
from collections.abc import Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from copilot.agent.base import Tool, ToolResult
from copilot.db.models import Chunk, Document, Image
from copilot.retrieval.base import Evidence, EvidenceKind, Retriever
from copilot.retrieval.image_retriever import ImageRetriever

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], Session] | sessionmaker


# search_documents, search_images


class SearchDocumentsTool(Tool):
    name = "search_documents"
    description = "Semantic search over manual TEXT for passages relevant to a query."
    parameters = {
        "query": "the search text (required)",
        "document_id": "restrict the search to one manual (optional)",
        "top_k": "how many passages to return (optional)",
    }

    def __init__(self, retriever: Retriever, default_top_k: int = 5) -> None:
        self.retriever = retriever
        self.default_top_k = default_top_k

    def run(
        self,
        query: str,
        document_id: str | None = None,
        top_k: int | None = None,
        **_: object,
    ) -> ToolResult:
        results = self.retriever.retrieve(
            query, top_k=top_k or self.default_top_k, document_id=document_id
        )
        return ToolResult(tool_name=self.name, output=results)


class SearchImagesTool(Tool):
    name = "search_images"
    description = (
        "Semantic search over manual DIAGRAMS and PHOTOS for images relevant to a query. "
        "Use this when the question is specifically about a picture, diagram, or visual."
    )
    parameters = {
        "query": "the search text (required)",
        "document_id": "restrict the search to one manual (optional)",
        "top_k": "how many images to return (optional)",
    }

    def __init__(self, image_retriever: ImageRetriever | None, default_top_k: int = 5) -> None:
        self.image_retriever = image_retriever
        self.default_top_k = default_top_k

    def run(
        self,
        query: str,
        document_id: str | None = None,
        top_k: int | None = None,
        **_: object,
    ) -> ToolResult:
        if self.image_retriever is None:
            # no image model loaded, nothing to search with
            return ToolResult(tool_name=self.name, output=[])
        results = self.image_retriever.retrieve(
            query, top_k=top_k or self.default_top_k, document_id=document_id
        )
        return ToolResult(tool_name=self.name, output=results)


# get_page


class PageNotFoundError(ValueError):
    pass


class GetPageTool(Tool):
    name = "get_page"
    description = (
        "Fetch the exact text and any diagrams from ONE specific page of a manual, by "
        "page number. Use this when a question names a page, or when a previous search "
        "result's page needs to be read in full rather than as a single retrieved passage."
    )
    parameters = {
        "document_id": "id of the manual (required)",
        "page_number": "the page to fetch (required)",
    }

    def __init__(self, session_factory: SessionFactory) -> None:
        self.session_factory = session_factory

    def run(self, document_id: str, page_number: int, **_: object) -> ToolResult:
        # page_number may arrive as a string from parsed json, coercing early
        try:
            page_number = int(page_number)
        except (TypeError, ValueError) as error:
            raise PageNotFoundError(f"page_number must be an integer, got {page_number!r}") from error

        session = self.session_factory()
        try:
            chunks = list(
                session.scalars(
                    select(Chunk)
                    .where(Chunk.document_id == document_id, Chunk.page_number == page_number)
                    .order_by(Chunk.chunk_index)
                )
            )
            images = list(
                session.scalars(
                    select(Image)
                    .where(Image.document_id == document_id, Image.page_number == page_number)
                    .order_by(Image.image_index)
                )
            )
        finally:
            session.close()

        if not chunks and not images:
            raise PageNotFoundError(f"No content found for document {document_id} page {page_number}")

        # built from each row's own columns, reflecting verified db state
        evidence: list[Evidence] = [
            Evidence(
                kind=EvidenceKind.TEXT,
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                # exact lookup, not a ranked result
                score=1.0,
                chunk_id=chunk.id,
                text=chunk.text,
            )
            for chunk in chunks
        ]
        evidence.extend(
            Evidence(
                kind=EvidenceKind.IMAGE,
                document_id=image.document_id,
                page_number=image.page_number,
                score=1.0,
                image_id=image.id,
                image_path=image.storage_path,
                text=image.caption,
            )
            for image in images
        )
        return ToolResult(tool_name=self.name, output=evidence)


# get_document_metadata


class GetDocumentMetadataTool(Tool):
    name = "get_document_metadata"
    description = (
        "List the manuals that have been uploaded (id, filename, page count, status), or "
        "look up one manual's metadata by id. Use this for questions about which manuals "
        "exist, or how many pages one has — not for questions about their content."
    )
    parameters = {"document_id": "look up one manual by id (optional; omit to list all)"}

    def __init__(self, session_factory: SessionFactory) -> None:
        self.session_factory = session_factory

    def run(self, document_id: str | None = None, **_: object) -> ToolResult:
        session = self.session_factory()
        try:
            if document_id:
                document = session.get(Document, document_id)
                documents = [document] if document is not None else []
            else:
                documents = list(
                    session.scalars(select(Document).order_by(Document.uploaded_at.desc()))
                )

            output = []
            for document in documents:
                chunk_count = (
                    session.scalar(
                        select(func.count()).select_from(Chunk).where(Chunk.document_id == document.id)
                    )
                    or 0
                )
                image_count = (
                    session.scalar(
                        select(func.count()).select_from(Image).where(Image.document_id == document.id)
                    )
                    or 0
                )
                output.append(
                    {
                        "id": document.id,
                        "filename": document.filename,
                        "status": document.status,
                        "page_count": document.page_count,
                        "chunk_count": chunk_count,
                        "image_count": image_count,
                    }
                )
        finally:
            session.close()

        return ToolResult(tool_name=self.name, output=output)


# calculate


class CalculatorError(ValueError):
    pass


# whitelist of arithmetic operators only, no names or calls reachable
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# caps a single pow node so a crafted expression cannot burn cpu forever
_MAX_EXPONENT_MAGNITUDE = 1000


def safe_eval(expression: str) -> float:
    """arithmetic without eval()"""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise CalculatorError(f"Could not parse expression: {expression!r}") from error

    try:
        result = _eval_node(tree.body)
    except ZeroDivisionError as error:
        raise CalculatorError(f"Division by zero in: {expression!r}") from error
    except OverflowError as error:
        raise CalculatorError(f"Result too large to represent: {expression!r}") from error

    return float(result)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError(f"Unsupported value in expression: {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"Unsupported operator: {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT_MAGNITUDE:
            raise CalculatorError(f"Exponent too large: {right}")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))

    raise CalculatorError(f"Unsupported expression syntax: {type(node).__name__}")


class CalculatorTool(Tool):
    name = "calculate"
    description = (
        "Evaluate an arithmetic expression, e.g. '(95-80)/80*100' for a percentage "
        "difference. Always use this for arithmetic rather than computing it yourself."
    )
    parameters = {"expression": "an arithmetic expression (required)"}

    def run(self, expression: str, **_: object) -> ToolResult:
        return ToolResult(tool_name=self.name, output=safe_eval(expression))
