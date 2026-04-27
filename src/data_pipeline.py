"""
Data Pipeline: ingestion, cleaning, chunking.
"""

from dataclasses import dataclass

@dataclass
class Document:
    """A single document with metadata."""
    content: str
    source: str
    title: str = ""
    doc_type: str = ""  # "tutorial" | "api_reference" | "guide"

@dataclass
class Chunk:
    """A single chunk with metadata for retrieval."""
    text: str
    source: str
    title: str = ""
    doc_type: str = ""
    chunk_index: int = 0

class DataPipeline:
    """Handles document ingestion, cleaning, and chunking."""
    def __init__(self, config):
        self.config = config

    def ingest(self, source_urls: list[str]) -> list[Document]:
        """Fetch documents from URLs or local files."""
        pass

    def clean(self, doc: Document) -> Document:
        """Remove navigation, footer, sidebar, etc."""
        pass

    def chunk(self, doc: Document) -> list[Chunk]:
        """
        Split document into chunks with overlap.
        Day 30: implement with heading-aware splitting
        Use config.chunk_size, config.chunk_overlap
        """
        pass

    def run(self, source_url: str) -> list[Chunk]:
        """End to end: ingest -> clean-> chunk all documents.
        """
        pass

