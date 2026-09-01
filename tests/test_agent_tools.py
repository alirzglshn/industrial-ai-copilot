from pathlib import Path

from PIL import Image as PILImage
from sqlalchemy.orm import Session, sessionmaker

from copilot.agent.tools import (
    GetDocumentMetadataTool,
    GetPageTool,
    PageNotFoundError,
    SearchDocumentsTool,
    SearchImagesTool,
)
from copilot.db.models import Chunk, Document, Image
from copilot.retrieval.base import EvidenceKind
from copilot.retrieval.deps import RetrievalStack
from tests.fakes import HashingEmbedder, HashingImageEmbedder


def _document(db: Session, document_id: str = "doc-a", filename: str = "manual.pdf") -> Document:
    document = Document(id=document_id, filename=filename, status="indexed", page_count=5)
    db.add(document)
    db.commit()
    return document


# --- SearchDocumentsTool ----------------------------------------------------


class TestSearchDocumentsTool:
    def test_finds_indexed_text(
        self, retrieval_stack: RetrievalStack, embedder: HashingEmbedder
    ) -> None:
        retrieval_stack.store.upsert(
            ids=["11111111-1111-1111-1111-111111111111"],
            vectors=embedder.embed_documents(["cooling airflow overheating"]),
            payloads=[
                {"document_id": "doc-a", "page_number": 4, "chunk_index": 0, "text": "cooling airflow overheating"}
            ],
        )
        tool = SearchDocumentsTool(retrieval_stack.retriever)

        result = tool.run(query="cooling airflow")

        assert result.tool_name == "search_documents"
        assert result.output
        assert result.output[0].kind is EvidenceKind.TEXT
        assert result.output[0].page_number == 4

    def test_respects_top_k(self, retrieval_stack: RetrievalStack, embedder: HashingEmbedder) -> None:
        retrieval_stack.store.upsert(
            ids=[f"1111111{i}-1111-1111-1111-11111111111{i}" for i in range(5)],
            vectors=embedder.embed_documents([f"cooling airflow text {i}" for i in range(5)]),
            payloads=[
                {"document_id": "doc-a", "page_number": i, "chunk_index": i, "text": f"cooling airflow text {i}"}
                for i in range(5)
            ],
        )
        tool = SearchDocumentsTool(retrieval_stack.retriever, default_top_k=5)

        assert len(tool.run(query="cooling", top_k=2).output) == 2

    def test_document_id_scopes_the_search(
        self, retrieval_stack: RetrievalStack, embedder: HashingEmbedder
    ) -> None:
        retrieval_stack.store.upsert(
            ids=["22222222-2222-2222-2222-222222222222", "33333333-3333-3333-3333-333333333333"],
            vectors=embedder.embed_documents(["cooling airflow", "cooling airflow"]),
            payloads=[
                {"document_id": "doc-a", "page_number": 1, "chunk_index": 0, "text": "cooling airflow"},
                {"document_id": "doc-b", "page_number": 1, "chunk_index": 0, "text": "cooling airflow"},
            ],
        )
        tool = SearchDocumentsTool(retrieval_stack.retriever)

        results = tool.run(query="cooling airflow", document_id="doc-b").output

        assert {item.document_id for item in results} == {"doc-b"}


# --- SearchImagesTool -------------------------------------------------------


class TestSearchImagesTool:
    def test_finds_indexed_images(
        self,
        retrieval_stack: RetrievalStack,
        image_embedder: HashingImageEmbedder,
        tmp_path: Path,
    ) -> None:
        image_path = tmp_path / "impeller.png"
        PILImage.new("RGB", (80, 80), color=(10, 120, 200)).save(image_path)
        image_embedder.describe(str(image_path), "impeller diagram")
        retrieval_stack.image_store.upsert(
            ids=["44444444-4444-4444-4444-444444444444"],
            vectors=image_embedder.embed_images([str(image_path)]),
            payloads=[
                {
                    "document_id": "doc-a",
                    "page_number": 12,
                    "storage_path": str(image_path),
                    "caption": "impeller diagram",
                }
            ],
        )
        tool = SearchImagesTool(retrieval_stack.image_retriever)

        result = tool.run(query="impeller diagram")

        assert result.tool_name == "search_images"
        assert result.output
        assert result.output[0].kind is EvidenceKind.IMAGE
        assert result.output[0].page_number == 12

    def test_returns_empty_when_image_retriever_is_unavailable(self) -> None:
        tool = SearchImagesTool(image_retriever=None)

        result = tool.run(query="anything")

        assert result.output == []


# --- GetPageTool -------------------------------------------------------------


class TestGetPageTool:
    def test_returns_the_pages_text_and_images(self, db_session: Session, session_factory: sessionmaker) -> None:
        _document(db_session)
        db_session.add(Chunk(document_id="doc-a", page_number=3, chunk_index=0, text="first paragraph"))
        db_session.add(Chunk(document_id="doc-a", page_number=3, chunk_index=1, text="second paragraph"))
        db_session.add(Image(document_id="doc-a", page_number=3, image_index=0, storage_path="/img/p3.png"))
        db_session.commit()

        result = GetPageTool(session_factory).run(document_id="doc-a", page_number=3)

        assert result.tool_name == "get_page"
        text_items = [e for e in result.output if e.kind is EvidenceKind.TEXT]
        image_items = [e for e in result.output if e.kind is EvidenceKind.IMAGE]
        assert [e.text for e in text_items] == ["first paragraph", "second paragraph"]
        assert len(image_items) == 1
        assert image_items[0].image_path == "/img/p3.png"

    def test_only_returns_content_from_the_requested_page(
        self, db_session: Session, session_factory: sessionmaker
    ) -> None:
        _document(db_session)
        db_session.add(Chunk(document_id="doc-a", page_number=3, chunk_index=0, text="page three"))
        db_session.add(Chunk(document_id="doc-a", page_number=4, chunk_index=0, text="page four"))
        db_session.commit()

        result = GetPageTool(session_factory).run(document_id="doc-a", page_number=3)

        assert [e.text for e in result.output] == ["page three"]

    def test_does_not_leak_content_from_another_document(
        self, db_session: Session, session_factory: sessionmaker
    ) -> None:
        _document(db_session, "doc-a")
        _document(db_session, "doc-b")
        db_session.add(Chunk(document_id="doc-a", page_number=1, chunk_index=0, text="from A"))
        db_session.add(Chunk(document_id="doc-b", page_number=1, chunk_index=0, text="from B"))
        db_session.commit()

        result = GetPageTool(session_factory).run(document_id="doc-a", page_number=1)

        assert [e.text for e in result.output] == ["from A"]

    def test_page_number_arriving_as_a_string_is_coerced_to_int(
        self, db_session: Session, session_factory: sessionmaker
    ) -> None:
        """A model's JSON can emit "5" instead of 5.

        Evidence.page_number must come back as int, or citation matching in
        copilot.generation.grounding (which compares it against the int page
        numbers parsed from "[page N]") would treat a genuine citation as
        fabricated just because '5' != 5.
        """
        _document(db_session)
        db_session.add(Chunk(document_id="doc-a", page_number=5, chunk_index=0, text="page five text"))
        db_session.commit()

        result = GetPageTool(session_factory).run(document_id="doc-a", page_number="5")

        assert result.output[0].page_number == 5
        assert isinstance(result.output[0].page_number, int)

    def test_unparseable_page_number_raises_rather_than_reaching_the_query(
        self, db_session: Session, session_factory: sessionmaker
    ) -> None:
        _document(db_session)
        db_session.commit()

        try:
            GetPageTool(session_factory).run(document_id="doc-a", page_number="not-a-number")
            assert False, "expected PageNotFoundError"
        except PageNotFoundError:
            pass

    def test_missing_page_raises(self, db_session: Session, session_factory: sessionmaker) -> None:
        _document(db_session)
        db_session.commit()

        try:
            GetPageTool(session_factory).run(document_id="doc-a", page_number=99)
            assert False, "expected PageNotFoundError"
        except PageNotFoundError:
            pass


# --- GetDocumentMetadataTool --------------------------------------------------


class TestGetDocumentMetadataTool:
    def test_lists_all_documents_when_no_id_given(
        self, db_session: Session, session_factory: sessionmaker
    ) -> None:
        _document(db_session, "doc-a", "first.pdf")
        _document(db_session, "doc-b", "second.pdf")
        db_session.commit()

        result = GetDocumentMetadataTool(session_factory).run()

        filenames = {row["filename"] for row in result.output}
        assert filenames == {"first.pdf", "second.pdf"}

    def test_looks_up_one_document_by_id(self, db_session: Session, session_factory: sessionmaker) -> None:
        _document(db_session, "doc-a", "manual.pdf")
        db_session.commit()

        result = GetDocumentMetadataTool(session_factory).run(document_id="doc-a")

        assert len(result.output) == 1
        assert result.output[0]["filename"] == "manual.pdf"
        assert result.output[0]["page_count"] == 5

    def test_includes_chunk_and_image_counts(self, db_session: Session, session_factory: sessionmaker) -> None:
        _document(db_session)
        db_session.add(Chunk(document_id="doc-a", page_number=1, chunk_index=0, text="a"))
        db_session.add(Chunk(document_id="doc-a", page_number=1, chunk_index=1, text="b"))
        db_session.add(Image(document_id="doc-a", page_number=1, image_index=0, storage_path="/x.png"))
        db_session.commit()

        row = GetDocumentMetadataTool(session_factory).run(document_id="doc-a").output[0]

        assert row["chunk_count"] == 2
        assert row["image_count"] == 1

    def test_unknown_document_id_returns_empty_list(
        self, db_session: Session, session_factory: sessionmaker
    ) -> None:
        result = GetDocumentMetadataTool(session_factory).run(document_id="does-not-exist")

        assert result.output == []
