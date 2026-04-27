"""
Data Pipeline: ingestion, cleaning, chunking.
"""

import re, time, json, logging, requests
from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """A single document with metadata."""
    content: str
    source: str
    title: str = ""
    doc_type: str = ""  # "tutorial" | "api_reference" | "guide"
    url: str = ""

@dataclass
class Chunk:
    """A single chunk with metadata for retrieval."""
    text: str
    source: str
    title: str = ""
    doc_type: str = ""
    heading_path: str = ""
    chunk_index: int = 0

class DataPipeline:
    """Handles document ingestion, cleaning, and chunking."""
    PRIORITY_DOCS = [
        "quicktour.md", "training.md", "trainer.md",
        "tasks/image_classification.md", "tokenizer_summary.md",
        "peft.md", "model_sharing.md",
        "installation.md", "tasks/sequence_classification.md",
        "tasks/token_classification.md",
        "tasks/question_answering.md",
        "tasks/summarization.md", "tasks/language_modeling.md",
        "generation_strategies.md", "llm_tutorial.md",
        "pipeline_tutorial.md", "create_a_model.md",
        "run_scripts.md",
    ]

    GITHUB_RAW = (
        "https://raw.githubusercontent.com/"
        "huggingface/transformers/main/docs/source/en"
    )
    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config.raw_data_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def ingest(self, source_urls=None) -> list[Document]:
        """Fetch documents from URLs or local files."""
        docs = []
        targets = source_urls or self.PRIORITY_DOCS

        for filename in tqdm(targets, desc="Downloading"):
            local_path = self.raw_dir / filename
            if local_path.exists():
                logger.info(f"Cache hit: {filename}")
                content = local_path.read_text(encoding="utf-8")
            else:
                url = f"{self.GITHUB_RAW}/{filename}"
                try:
                    resp = requests.get(url, timeout=15)
                    resp.raise_for_status()
                    content = resp.text
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_text(content, encoding="utf-8")
                    time.sleep(0.3)
                except requests.RequestException as e:
                    logger.warning(f"Failed: {filename}: {e}")
                    continue

            doc_type = self._classify_doc(filename, content)
            title = self._extract_title(content) or filename
            docs.append(Document(
                content=content, source=filename,
                title=title, doc_type=doc_type,
                url=f"{self.GITHUB_RAW}/{filename}",
            ))

        logger.info(f"Ingested {len(docs)} documents")
        return docs

    def _classify_doc(self, filename, content):
        if "tasks/" in filename: return "tutorial"
        if "tutorial" in filename: return "tutorial"
        if "[[autodoc]]" in content: return "api_reference"
        return "guide"

    def _extract_title(self, content):
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        return match.group(1).strip() if match else ""

    def clean(self, doc):
        """Remove noise from markdown document."""
        text = doc.content

        # 1. YAML front matter
        text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)

        # 2. autodoc tags
        text = re.sub(r"\[\[autodoc\]\].*?(?=\n#|\n\[\[|$)", "", text, flags=re.DOTALL)

        # 3. doc-builder tags: [[open-in-colab]] etc.
        text = re.sub(r"\[\[[^\]]+\]\]\n?", "", text)

        # 4. HTML comments (includes copyright blocks)
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

        # 5. HF custom tags (opening, closing, and self-closing)
        text = re.sub(
            r"</?(?:Tip|frameworkcontent|jax|tf|pt|Deprecated"
            r"|hfoptions|hfoption|Youtube)(?:\s[^>]*)?>",
            "", text
        )

        # 6. iframe blocks (dataset previews)
        text = re.sub(r"<iframe[\s\S]*?</iframe>", "", text)

        # 7. div+img blocks (images useless for RAG)
        text = re.sub(r"<div[^>]*>.*?</div>", "", text, flags=re.DOTALL)

        # 8. Simplify markdown links: [text](url) -> text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # 9. Collapse 3+ blank lines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        text = text.strip()
        return Document(
            content=text, source=doc.source,
            title=doc.title, doc_type=doc.doc_type, url=doc.url,
        )

    def run(self, source_urls: list[str] = None) -> list[Chunk]:
        """End to end: ingest -> clean-> chunk all documents.
        """
        docs = self.ingest(source_urls)
        all_chunks = []
        for doc in docs:
            cleaned = self.clean(doc)
            chunks = self.chunk(cleaned)
            all_chunks.extend(chunks)
        logger.info(f"Pipeline: {len(docs)} docs -> {len(all_chunks)}")
        self._save_chunks(all_chunks)
        return all_chunks

    def _save_chunks(self, chunks):
        """Save chunks to JSON for Day 31 Indexer"""
        output_dir = Path(self.config.processed_data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "chunks.json"
        data = [{"text": chunk.text, "source": chunk.source,
                 "title": chunk.title, "doc_type": chunk.doc_type,
                 "chunk_index": chunk.chunk_index,
                 "heading_path": chunk.heading_path}
                for chunk in chunks]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

    @staticmethod
    def load_chunks(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [Chunk(**item) for item in data]

    def chunk(self, doc: Document) -> list[Chunk]:
        """Heading-aware chunking for Markdown documents."""
        sections = self._split_by_headings(doc.content)
        sections = self._merge_short_sections(sections)
        chunks = []
        chunk_idx = 0

        for heading_path, section_text in sections:
            prefix = ""
            if heading_path:
                prefix = "[" + heading_path + "]\n\n"

            if len(prefix + section_text) <= self.config.chunk_size:
                chunk_text = prefix + section_text
                if len(chunk_text.strip()) >= self.config.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text, source=doc.source,
                        title=doc.title, doc_type=doc.doc_type,
                        chunk_index=chunk_idx, heading_path=heading_path,
                    ))
                    chunk_idx += 1
            else:
                max_body = self.config.chunk_size - len(prefix)
                sub_chunks = self._recursive_split(
                    section_text, max_body, self.config.chunk_overlap)
                for sub_text in sub_chunks:
                    chunk_text = prefix + sub_text
                    if len(chunk_text.strip()) >= self.config.min_chunk_size:
                        chunks.append(Chunk(
                            text=chunk_text, source=doc.source,
                            title=doc.title, doc_type=doc.doc_type,
                            chunk_index=chunk_idx, heading_path=heading_path,
                        ))
                        chunk_idx += 1

        logger.info(f"Chunked {doc.source}: {len(chunks)} chunks")
        return chunks

    def _split_by_headings(self, text):
        """Split markdown by headings -> (heading_path, content) pairs."""
        heading_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
        sections = []
        heading_stack = []
        last_end = 0

        for match in heading_re.finditer(text):
            if last_end > 0 or match.start() > 0:
                content = text[last_end:match.start()].strip()
                if content:
                    path = " > ".join(h[1] for h in heading_stack)
                    sections.append((path, content))
            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            last_end = match.end()

        remaining = text[last_end:].strip()
        if remaining:
            path = " > ".join(h[1] for h in heading_stack)
            sections.append((path, remaining))
        return sections

    def _recursive_split(self, text, max_size, overlap):
        """Split: paragraphs first, then sentences."""
        if len(text) <= max_size:
            return [text]
        parts = re.split(r"(\n\n)", text)
        chunks = []
        current = ""
        for part in parts:
            if len(current) + len(part) <= max_size:
                current += part
            else:
                if current.strip():
                    chunks.append(current.strip())
                if len(part) > max_size:
                    sentences = re.split(r"(?<=[.!?])\s+", part)
                    sub_current = ""
                    for sent in sentences:
                        if len(sub_current) + len(sent) <= max_size:
                            sub_current = (sub_current + " " + sent).strip()
                        else:
                            if sub_current.strip():
                                chunks.append(sub_current.strip())
                            sub_current = sent
                    current = sub_current if sub_current.strip() else ""
                else:
                    current = part
        if current.strip():
            chunks.append(current.strip())

        if overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                tail = chunks[i - 1][-overlap:]
                sp = tail.find(" ")
                if sp != -1: tail = tail[sp + 1:]
                overlapped.append(tail + " " + chunks[i])
            chunks = overlapped

        final = []
        for c in chunks:
            if len(c) > max_size:
                for i in range(0, len(c), max_size):
                    final.append(c[i:i + max_size])
            else:
                final.append(c)
        chunks = final

        return chunks

    def _protect_code_blocks(self, text):
        """Replace fenced code blocks with placeholders."""
        code_blocks = {}
        counter = [0]

        def replacer(match):
            key = f"__CODEBLOCK_{counter[0]}__"
            code_blocks[key] = match.group(0)
            counter[0] += 1
            return key

        # match triple-backtick fences
        protected = re.sub(r"\x60\x60\x60[\s\S]*?\x60\x60\x60", replacer, text)
        return protected, code_blocks

    def _restore_code_blocks(self, text, code_blocks):
        for key, block in code_blocks.items():
            text = text.replace(key, block)
        return text

    def _merge_short_sections(self, sections, min_size=200):
        if not sections: return sections
        merged = [sections[0]]
        for path, content in sections[1:]:
            prev_path, prev_content = merged[-1]
            if (len(prev_content) < min_size and len(content) < min_size
                    and self._share_parent(prev_path, path)):
                merged[-1] = (path, prev_content + "\n\n" + content)
            else:
                merged.append((path, content))
        return merged

    def _share_parent(self, path1, path2):
        return path1.split(" > ")[:-1] == path2.split(" > ")[:-1]