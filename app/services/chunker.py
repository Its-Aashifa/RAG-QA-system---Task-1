"""
Text chunking service.

Strategy: Sentence-aware sliding window chunking.

Why not naive fixed-size splits?
  - Splitting mid-sentence breaks semantic units, hurting embedding quality.
  - We split on sentence boundaries first, then group into chunks of ~CHUNK_SIZE chars
    with CHUNK_OVERLAP chars of overlap between consecutive chunks.

See docs/design_decisions.md for detailed rationale on chunk size choice.
"""

import re
import logging
from dataclasses import dataclass
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    source: str
    char_start: int
    char_end: int


def split_into_sentences(text: str) -> list[str]:
    """
    Naive but effective sentence splitter.
    Splits on '.', '!', '?' followed by whitespace or end-of-string.
    Preserves the delimiter at the end of each sentence.
    """
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty/whitespace-only sentences
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, source: str) -> list[TextChunk]:
    """
    Chunk text using a sentence-aware sliding window.

    Algorithm:
    1. Split text into sentences.
    2. Accumulate sentences until chunk_size is reached.
    3. On overflow, save the chunk and backtrack by overlap amount.
    4. This ensures overlapping context between consecutive chunks.

    Args:
        text: Full document text.
        source: Filename/identifier for provenance tracking.

    Returns:
        List of TextChunk objects.
    """
    chunk_size = settings.CHUNK_SIZE
    chunk_overlap = settings.CHUNK_OVERLAP

    sentences = split_into_sentences(text)
    if not sentences:
        logger.warning(f"No sentences extracted from '{source}'")
        return []

    chunks: list[TextChunk] = []
    current_sentences: list[str] = []
    current_len = 0
    char_cursor = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence would exceed chunk_size, flush current chunk
        if current_len + sentence_len > chunk_size and current_sentences:
            chunk_text_str = " ".join(current_sentences)
            chunks.append(TextChunk(
                text=chunk_text_str,
                chunk_index=chunk_index,
                source=source,
                char_start=char_cursor - current_len,
                char_end=char_cursor,
            ))
            chunk_index += 1

            # Backtrack: keep tail sentences whose total length <= overlap
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break

            current_sentences = overlap_sentences
            current_len = overlap_len

        current_sentences.append(sentence)
        current_len += sentence_len
        char_cursor += sentence_len + 1  # +1 for the space between sentences

    # Flush remaining sentences as the final chunk
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        chunks.append(TextChunk(
            text=chunk_text_str,
            chunk_index=chunk_index,
            source=source,
            char_start=char_cursor - current_len,
            char_end=char_cursor,
        ))

    logger.info(f"'{source}' → {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
