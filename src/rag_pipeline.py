"""End-to-end RAG Pipeline: teis all modules together."""
from config import RAGConfig
from src.data_pipeline import DataPipeline
from src.indexer import Indexer
from src.retriever import Retriever
from src.generator import Generator


class RAGPipeline:
    """Production RAG: retrieve -> generate -> return.
    Orchestrates all modules. Each modul only know its own layer:
    - DataPipeline: document -> chunks (does not know about indexer)
    - Indexer: chunks -> index (does not know where chunks came from)
    - Retriever: query -> relevant chunks
    - Generator: query + contexts -> answer
    """

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.data_pipeline = DataPipeline(self.config)
        self.indexer = Indexer(self.config)
        self.retriever = Retriever(self.config, self.indexer)
        self.generator = Generator(self.config)

    def build_index(self, source_urls: list[str]) -> None:
        """Full index build: ingest -> clean -> chunk -> index all documents."""
        chunks = self.data_pipeline.run(source_urls)
        self.indexer.build(chunks)

    def add_document(self, url: str) -> None:
        """Incrementally add a new document: ingest -> clean -> chunk -> add to index."""
        chunks = self.data_pipeline.run([url])
        self.indexer.add_chunks(chunks)

    def remove_document(self, source: str) -> None:
        """Remove a document's chunks from the index."""
        self.indexer.delete_by_source(source)

    def query(self, question: str):
        """Answer a question using RAG.
        Returns: {
            "answer": str,
            "contexts": list[dict],
            "metadata": dict
        }
        """
        # Step 1: Retrieve relevant chunks
        contexts = self.retriever.retrieve(question)

        # Step 2: Generate answer from contexts
        result = self.generate.answer(question, contexts)

        return {
            "answer": result.get("answer", ""),
            "contexts": contexts,
            "metadata": {},
        }
