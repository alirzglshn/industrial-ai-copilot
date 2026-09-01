"""splitting extracted page text into retrievable, page-scoped chunks"""

import re
from dataclasses import dataclass

from copilot.ingestion.base import ParsedDocument

TABLE_PREFIX = "[Table]"

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TextChunk:
    """a unit of text to embed and retrieve, traceable to one source page"""

    page_number: int
    chunk_index: int
    text: str


def _hard_split(text: str, max_len: int) -> list[str]:
    return [text[i : i + max_len] for i in range(0, len(text), max_len)]


def _segment(text: str, max_len: int) -> list[str]:
    """breaking text into pieces no longer than max_len, preferring natural boundaries"""
    segments: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_len:
            segments.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= max_len:
                segments.append(sentence)
            else:
                segments.extend(_hard_split(sentence, max_len))
    return segments


def _overlap_tail(segments: list[str], overlap: int) -> list[str]:
    """trailing segments of a finished chunk to repeat at the start of the next"""
    if overlap <= 0:
        return []
    tail: list[str] = []
    total = 0
    for segment in reversed(segments):
        addition = len(segment) + (1 if tail else 0)
        if total + addition > overlap:
            break
        tail.insert(0, segment)
        total += addition
    return tail


class TextChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        # overlap kept below chunk_size or the splitter would stop progressing
        self.chunk_size = chunk_size
        self.chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    def _pack(self, segments: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for segment in segments:
            addition = len(segment) + (1 if current else 0)
            if current and current_len + addition > self.chunk_size:
                chunks.append("\n".join(current))
                current = _overlap_tail(current, self.chunk_overlap)
                current_len = sum(len(s) for s in current) + max(0, len(current) - 1)
                addition = len(segment) + (1 if current else 0)
            # forcing an oversized segment in guarantees forward progress
            current.append(segment)
            current_len += addition

        if current:
            chunks.append("\n".join(current))
        return chunks

    def chunk_page(
        self,
        page_number: int,
        text: str,
        tables: list[str] | None = None,
        start_index: int = 0,
    ) -> list[TextChunk]:
        texts: list[str] = []
        if text and text.strip():
            texts.extend(self._pack(_segment(text, self.chunk_size)))
        for table in tables or []:
            if not table.strip():
                continue
            # prefixing every piece so an isolated chunk still reads as tabular
            for piece in self._pack(_segment(table, self.chunk_size - len(TABLE_PREFIX) - 1)):
                texts.append(f"{TABLE_PREFIX}\n{piece}")

        return [
            TextChunk(page_number=page_number, chunk_index=start_index + offset, text=chunk_text)
            for offset, chunk_text in enumerate(texts)
        ]

    def chunk_document(self, document: ParsedDocument) -> list[TextChunk]:
        """chunking every page, numbering chunks sequentially across the document"""
        chunks: list[TextChunk] = []
        for page in document.pages:
            chunks.extend(
                self.chunk_page(
                    page_number=page.page_number,
                    text=page.text,
                    tables=page.tables,
                    start_index=len(chunks),
                )
            )
        return chunks
