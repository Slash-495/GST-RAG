"""
GST Document Ingestion Pipeline
------------------------------
Extracts text from official GST PDFs (Acts, Rules, Notifications), chunks them
specifically by legal sections and chapters, generates dense vector embeddings
using Google Generative AI (Gemini text-embedding-004), and persists them into
a local FAISS index with rich metadata.
"""

import os
import sys
import re
import json
import pickle
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pymupdf
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("gst_ingest")


def load_api_key() -> str:
    """Load Gemini API Key from environment or .env file."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning(
            "GEMINI_API_KEY / GOOGLE_API_KEY not found in environment or .env file. "
            "Embeddings will fail unless provided via --api-key or in .env."
        )
    return api_key or ""


def extract_pages_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF file page by page using PyMuPDF.
    
    Returns:
        List of dicts: [{"page_number": int, "text": str}]
    """
    pages_data = []
    try:
        doc = pymupdf.open(pdf_path)
        logger.info(f"Loaded '{pdf_path.name}' with {len(doc)} pages.")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            pages_data.append({
                "page_number": page_num + 1,
                "text": text
            })
        doc.close()
    except Exception as e:
        logger.error(f"Failed to read PDF '{pdf_path}': {e}")
        raise
    return pages_data


class LegalSectionChunker:
    """
    Chunks legal text (specifically GST Acts, Rules, and Notifications) by legal sections.
    Detects Chapters, Sections (both single-line and two-line formats), GST Forms, Schedules,
    and Service Headings, while filtering running headers and cross-references.
    """

    CHAPTER_PATTERN = re.compile(
        r'^\s*(?:CHAPTER\s+([IVXLCDM\d]+))\b\s*[:\.\-]?\s*(.*)$',
        re.IGNORECASE
    )

    SCHEDULE_PATTERN = re.compile(
        r'^\s*(SCHEDULE\s+[IVXLCDM\d]+)\b\s*[:\.\-]?\s*([A-Z\s\,\(\)\/\-]{3,})?$',
        re.IGNORECASE
    )

    ORDER_PATTERN = re.compile(
        r'^\s*(Order\s+No\.\s*[\w\/\-]+)\b\s*(.*)$',
        re.IGNORECASE
    )

    FORM_PATTERN = re.compile(
        r'^\s*(FORM\s+GST\s+[A-Z0-9\-\(\)]+)\b\s*(.*)$',
        re.IGNORECASE
    )

    SERVICE_HEADING_PATTERN = re.compile(
        r'^\s*(Heading\s+\d{4})\b\s*(.*)$',
        re.IGNORECASE
    )

    EXPLICIT_SECTION_PATTERN = re.compile(
        r'^\s*Section\s+(\d+[A-Z]?)\.?\s*[\:\-\—]\s*([A-Z][A-Za-z\s\,\(\)\/\-]+)$'
    )

    BARE_ACT_ONE_LINE = re.compile(
        r'^\s*(\d+[A-Z]?)\.\s+([A-Z][a-zA-Z\s\,\(\)\/\-]{2,80}?)\s*(?:[—\-\:\.]|\s*\(1\))\s*(.*)$'
    )

    BARE_ACT_NUM_ONLY = re.compile(r'^\s*(\d+[A-Z]?)\.\s*$')
    BARE_ACT_TITLE_LINE = re.compile(r'^\s*([A-Z][a-zA-Z\s\,\(\)\/\-]{2,80}?)\s*(?:[—\-\:\.]|\s*\(1\))\s*(.*)$')

    INVALID_TITLE_STARTS = (
        'of ', 'and ', 'or ', 'to ', 'in ', 'for ', 'by ', 'with ',
        'under ', 'as ', 'the ', 'that ', 'from ', 'where ', 'if ', 'such '
    )

    def __init__(self, max_chunk_chars: int = 2500, chunk_overlap_chars: int = 300):
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars

    def parse_document(self, pages_data: List[Dict[str, Any]], source_filename: str) -> List[Dict[str, Any]]:
        """
        Parses pages into legal sections.
        """
        all_lines: List[Tuple[str, int, bool]] = []
        toc_pages = set()

        # Identify Table of Contents pages
        for p in pages_data:
            p_text = p["text"]
            if re.search(r'^\s*(?:table of contents|contents)\b', p_text, re.IGNORECASE | re.MULTILINE):
                toc_pages.add(p["page_number"])

        for p in pages_data:
            p_num = p["page_number"]
            is_toc_page = p_num in toc_pages
            lines = p["text"].splitlines()
            for l_idx, line in enumerate(lines):
                is_skip_line = False
                stripped = line.strip()

                # Running headers in top 4 lines of page (e.g. 'CHAPTER I PRELIMINARY')
                if l_idx < 4 and re.match(r'^\s*CHAPTER\s+[IVXLCDM\d]+\s+[A-Z\s]+$', stripped):
                    is_skip_line = True
                # Standalone page numbers in top 3 lines
                elif l_idx < 3 and re.match(r'^\s*\d+\s*$', stripped):
                    is_skip_line = True
                # Ignore lines with dotted table of contents: "FORM GST REG-01 .... 12"
                elif re.search(r'\.{3,}\s*\d+\s*$', stripped):
                    is_skip_line = True
                # Ignore standalone TOC heading
                elif re.match(r'^\s*(?:table of contents|contents)\s*$', stripped, re.IGNORECASE):
                    is_skip_line = True

                all_lines.append((line, p_num, is_skip_line))

        current_chapter = "PRELIMINARY"
        sections: List[Dict[str, Any]] = []
        current_sec: Optional[Dict[str, Any]] = None

        def start_new_section(sec_id: str, title: str, initial_text: str, page_num: int):
            nonlocal current_sec
            if current_sec:
                current_sec["text"] = "\n".join(current_sec["lines"]).strip()
                del current_sec["lines"]
                if current_sec["text"]:
                    sections.append(current_sec)

            clean_title = re.sub(r'[\.\:\;\—\-\s]+$', '', title).strip()
            current_sec = {
                "source": source_filename,
                "chapter": current_chapter,
                "section": sec_id,
                "title": clean_title if clean_title else sec_id,
                "pages": [page_num],
                "lines": [initial_text] if initial_text else []
            }

        i = 0
        n = len(all_lines)
        while i < n:
            line, page_num, is_skip = all_lines[i]
            stripped = line.strip()

            if is_skip or not stripped:
                if not is_skip and current_sec and current_sec["lines"]:
                    current_sec["lines"].append("")
                i += 1
                continue

            # Check Chapter
            chap_match = self.CHAPTER_PATTERN.match(stripped)
            if chap_match:
                chap_num = chap_match.group(1).strip()
                chap_title = chap_match.group(2).strip()
                if not chap_title and i + 1 < n:
                    next_line = all_lines[i+1][0].strip()
                    if next_line and not re.match(r'^\d+$', next_line) and not self.BARE_ACT_NUM_ONLY.match(next_line):
                        chap_title = next_line
                        i += 1
                current_chapter = f"CHAPTER {chap_num}" + (f": {chap_title}" if chap_title else "")
                i += 1
                continue

            # Check Removal of Difficulty Order (e.g. Order No. 1/2018-Central Tax)
            order_match = self.ORDER_PATTERN.match(stripped)
            if order_match:
                o_id = order_match.group(1).strip()
                o_title = order_match.group(2).strip()
                start_new_section(
                    sec_id=o_id,
                    title=o_title if o_title else o_id,
                    initial_text=line,
                    page_num=page_num
                )
                i += 1
                continue

            # Check Form (e.g. FORM GST REG-01)
            form_match = self.FORM_PATTERN.match(stripped)
            if form_match:
                f_id = form_match.group(1).strip()
                f_title = form_match.group(2).strip()
                start_new_section(
                    sec_id=f_id,
                    title=f_title if f_title else f_id,
                    initial_text=line,
                    page_num=page_num
                )
                i += 1
                continue

            # Check Schedule (e.g. SCHEDULE I)
            sched_match = self.SCHEDULE_PATTERN.match(stripped)
            if sched_match:
                s_id = sched_match.group(1).strip()
                s_title = sched_match.group(2).strip() if sched_match.group(2) else ""
                start_new_section(
                    sec_id=s_id,
                    title=s_title if s_title else s_id,
                    initial_text=line,
                    page_num=page_num
                )
                i += 1
                continue

            # Check Service Classification Heading (e.g. Heading 9961)
            service_match = self.SERVICE_HEADING_PATTERN.match(stripped)
            if service_match:
                h_id = service_match.group(1).strip()
                h_title = service_match.group(2).strip()
                start_new_section(
                    sec_id=h_id,
                    title=h_title if h_title else h_id,
                    initial_text=line,
                    page_num=page_num
                )
                i += 1
                continue

            # Check Explicit Section (e.g. Section 12 - Time of Supply)
            exp_match = self.EXPLICIT_SECTION_PATTERN.match(stripped)
            if exp_match:
                s_num = exp_match.group(1).strip()
                s_title = exp_match.group(2).strip()
                if not s_title.lower().startswith(self.INVALID_TITLE_STARTS) and len(s_title) > 2:
                    start_new_section(
                        sec_id=f"Section {s_num}",
                        title=s_title,
                        initial_text=line,
                        page_num=page_num
                    )
                    i += 1
                    continue

            # Check Two-line Bare Act format: "7.\nScope of supply.— (1)..."
            two_match = self.BARE_ACT_NUM_ONLY.match(stripped)
            if two_match and i + 1 < n:
                s_num = two_match.group(1).strip()
                next_l = all_lines[i+1][0].strip()
                t_match = self.BARE_ACT_TITLE_LINE.match(next_l)
                if t_match:
                    title = t_match.group(1).strip()
                    remaining = t_match.group(2).strip()
                    if not title.lower().startswith(self.INVALID_TITLE_STARTS) and len(title) > 2:
                        start_new_section(
                            sec_id=f"Section {s_num}",
                            title=title,
                            initial_text=remaining if remaining else next_l,
                            page_num=page_num
                        )
                        i += 2
                        continue

            # Check One-line Bare Act format: "1. Short title, extent...—(1)..."
            one_match = self.BARE_ACT_ONE_LINE.match(stripped)
            if one_match:
                s_num = one_match.group(1).strip()
                title = one_match.group(2).strip()
                remaining = one_match.group(3).strip()
                if not title.lower().startswith(self.INVALID_TITLE_STARTS) and len(title) > 2:
                    start_new_section(
                        sec_id=f"Section {s_num}",
                        title=title,
                        initial_text=remaining if remaining else line,
                        page_num=page_num
                    )
                    i += 1
                    continue

            # Regular line within active section
            if current_sec:
                current_sec["lines"].append(line)
                if page_num not in current_sec["pages"]:
                    current_sec["pages"].append(page_num)
            else:
                start_new_section(
                    sec_id="Preamble",
                    title="Preamble / Preliminary",
                    initial_text=line,
                    page_num=page_num
                )
            i += 1

        # Flush final section
        if current_sec:
            current_sec["text"] = "\n".join(current_sec["lines"]).strip()
            del current_sec["lines"]
            if current_sec["text"]:
                sections.append(current_sec)

        return self._subdivide_large_sections(sections)

    def _subdivide_large_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        If a section is larger than max_chunk_chars, intelligently subdivide it
        by sub-sections e.g. (1), (2), (3) or paragraphs, while prepending the parent
        legal section header to ensure retrieval context is retained.
        """
        final_chunks: List[Dict[str, Any]] = []

        for sec in sections:
            text = sec["text"]
            header = f"[{sec['source']} | {sec['chapter']} | {sec['section']}: {sec['title']}]"
            
            if len(text) <= self.max_chunk_chars:
                final_chunks.append({
                    "content": f"{header}\n\n{text}",
                    "raw_text": text,
                    "section": sec["section"],
                    "title": sec["title"],
                    "chapter": sec["chapter"],
                    "source": sec["source"],
                    "pages": sec["pages"],
                    "is_subchunk": False
                })
            else:
                # Subdivide by legal subsections e.g. "(1)", "(2)", or newlines
                paragraphs = re.split(r'\n(?=\s*\(\d+\)|\s*\([a-z]\)|\s*\n)', text)
                sub_blocks = []
                current_block = ""

                for p in paragraphs:
                    if len(current_block) + len(p) < self.max_chunk_chars:
                        current_block += ("\n" + p if current_block else p)
                    else:
                        if current_block:
                            sub_blocks.append(current_block)
                        current_block = p
                if current_block:
                    sub_blocks.append(current_block)

                for idx, block in enumerate(sub_blocks):
                    chunk_header = f"{header} (Part {idx + 1}/{len(sub_blocks)})"
                    final_chunks.append({
                        "content": f"{chunk_header}\n\n{block.strip()}",
                        "raw_text": block.strip(),
                        "section": sec["section"],
                        "title": sec["title"],
                        "chapter": sec["chapter"],
                        "source": sec["source"],
                        "pages": sec["pages"],
                        "is_subchunk": True,
                        "part": idx + 1,
                        "total_parts": len(sub_blocks)
                    })

        return final_chunks


import hashlib


def get_chunk_hash(chunk: Dict[str, Any]) -> str:
    """Generate a unique SHA-256 hash for a chunk based on source, section, and text."""
    key = f"{chunk.get('source', '')}|{chunk.get('section', '')}|{chunk.get('content', '')}"
    return hashlib.sha256(key.encode('utf-8')).hexdigest()


def generate_embeddings(
    chunks: List[Dict[str, Any]],
    api_key: str,
    model: str = "models/gemini-embedding-001",
    batch_size: int = 20,
    cache_path: Optional[Path] = None
) -> np.ndarray:
    """
    Generate embeddings for all text chunks using Google Generative AI.
    Features:
      - Persistent caching: Resumes from where it left off if interrupted.
      - Intelligent 429 rate-limit backoff: Parses exact delay requested by Gemini API.
      - Periodic atomic flushing to disk.
    """
    import google.generativeai as genai

    if not api_key:
        raise ValueError("Cannot embed chunks: Gemini API Key is missing.")

    genai.configure(api_key=api_key)

    # Load existing cache if available
    cache: Dict[str, List[float]] = {}
    if cache_path and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            logger.info(f"Loaded existing embedding cache from '{cache_path.name}' with {len(cache)} entries.")
        except Exception as e:
            logger.warning(f"Could not load cache from '{cache_path}': {e}. Starting fresh.")

    # Identify missing chunks
    missing_indices = []
    for idx, c in enumerate(chunks):
        h = get_chunk_hash(c)
        if h not in cache:
            missing_indices.append(idx)

    cached_count = len(chunks) - len(missing_indices)
    if cached_count > 0:
        logger.info(f"Cache hit: {cached_count}/{len(chunks)} chunks already embedded. Resuming for {len(missing_indices)} remaining chunks.")
    else:
        logger.info(f"Generating embeddings using '{model}' for all {len(chunks)} chunks...")

    # Check model availability and select fallback if needed
    active_model = model
    try:
        genai.embed_content(model=active_model, content="test", task_type="retrieval_document")
    except Exception as e:
        if "not found" in str(e).lower() and active_model != "models/gemini-embedding-001":
            logger.warning(f"Model '{active_model}' not found, falling back to 'models/gemini-embedding-001'")
            active_model = "models/gemini-embedding-001"

    if missing_indices:
        # Group missing chunks into batches
        missing_batches = [
            missing_indices[i:i + batch_size]
            for i in range(0, len(missing_indices), batch_size)
        ]

        for b_idx, batch_indices in enumerate(tqdm(missing_batches, desc="Embedding Batches")):
            batch_texts = [chunks[idx]["content"] for idx in batch_indices]
            max_retries = 10
            base_delay = 5

            for attempt in range(max_retries):
                try:
                    result = genai.embed_content(
                        model=active_model,
                        content=batch_texts,
                        task_type="retrieval_document"
                    )
                    embeddings = result["embedding"]

                    # Store in cache
                    for idx, emb in zip(batch_indices, embeddings):
                        h = get_chunk_hash(chunks[idx])
                        cache[h] = emb

                    # Flush cache atomically to disk after every batch
                    if cache_path:
                        temp_cache = cache_path.with_suffix(".tmp")
                        with open(temp_cache, "wb") as f:
                            pickle.dump(cache, f)
                        temp_cache.replace(cache_path)

                    break

                except Exception as e:
                    err_msg = str(e)
                    logger.warning(f"Batch {b_idx + 1}/{len(missing_batches)} (Attempt {attempt + 1}) encountered error: {err_msg}")

                    if attempt >= max_retries - 1:
                        logger.error(f"Failed to embed batch {b_idx + 1} after {max_retries} attempts.")
                        raise

                    # Check for rate limit / quota
                    if "429" in err_msg or "resourceexhausted" in err_msg.lower() or "quota" in err_msg.lower():
                        # Try to parse the suggested retry delay from the error message
                        retry_match = re.search(r'retry in\s+([\d\.]+)\s*s', err_msg, re.IGNORECASE)
                        sec_match = re.search(r'seconds:\s*(\d+)', err_msg)

                        if retry_match:
                            wait_seconds = float(retry_match.group(1)) + 5.0
                        elif sec_match:
                            wait_seconds = float(sec_match.group(1)) + 5.0
                        else:
                            wait_seconds = max(base_delay * (2 ** min(attempt, 4)), 45.0)

                        logger.info(f"Rate limit hit. Waiting {wait_seconds:.1f}s for quota window to reset...")
                        time.sleep(wait_seconds)
                    else:
                        wait_seconds = base_delay * (2 ** attempt)
                        logger.info(f"Retrying in {wait_seconds}s...")
                        time.sleep(wait_seconds)

            # Force a 15-second pause between API calls at the end of each batch iteration
            time.sleep(15)

    # Reconstruct the full embedding array in original chunk order
    all_embeddings = [cache[get_chunk_hash(c)] for c in chunks]
    emb_matrix = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"Embedding matrix ready with shape: {emb_matrix.shape}")
    return emb_matrix


def build_and_save_vectorstore(
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    output_dir: Path,
    index_name: str = "gst_index"
) -> Tuple[Path, Path]:
    """
    Normalizes embeddings, creates FAISS IndexFlatIP (Cosine similarity),
    and persists index alongside chunk metadata.
    """
    import faiss

    output_dir.mkdir(parents=True, exist_ok=True)
    dim = embeddings.shape[1]

    # L2-normalize vectors for Cosine Similarity using Inner Product
    normalized_embeddings = embeddings.copy()
    faiss.normalize_L2(normalized_embeddings)

    index = faiss.IndexFlatIP(dim)
    index.add(normalized_embeddings)
    logger.info(f"Created FAISS IndexFlatIP with {index.ntotal} vectors of dimension {dim}.")

    # Save index
    index_file = output_dir / f"{index_name}.faiss"
    faiss.write_index(index, str(index_file))
    logger.info(f"FAISS index saved to '{index_file}'.")

    # Prepare metadata
    metadata = {
        "index_name": index_name,
        "total_chunks": len(chunks),
        "dimension": dim,
        "created_at": time.time(),
        "chunks": chunks
    }

    # Save metadata as pickle (fast Python loading) and JSON (inspectable)
    pkl_file = output_dir / f"{index_name}_metadata.pkl"
    json_file = output_dir / f"{index_name}_metadata.json"

    with open(pkl_file, "wb") as f:
        pickle.dump(metadata, f)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Metadata saved to '{pkl_file}' and '{json_file}'.")
    return index_file, pkl_file


def process_gst_documents(
    raw_dir: Path,
    vectorstore_dir: Path,
    api_key: Optional[str] = None,
    index_name: str = "gst_index",
    model: str = "models/gemini-embedding-001",
    batch_size: int = 20,
    dry_run: bool = False
):
    """
    Complete pipeline to read raw GST PDFs, chunk them, embed them, and index in FAISS.
    """
    raw_dir = Path(raw_dir)
    vectorstore_dir = Path(vectorstore_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(list(raw_dir.glob("*.pdf")))
    if not pdf_files:
        logger.warning(f"No PDF files found in '{raw_dir}'.")
        print(f"\n[INFO] Please place official GST PDF documents in: {raw_dir.resolve()}")
        print("Example: CGST_Act_2017.pdf, IGST_Act_2017.pdf, etc.\n")
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s) in '{raw_dir}': {[f.name for f in pdf_files]}")

    chunker = LegalSectionChunker()
    all_chunks: List[Dict[str, Any]] = []

    for pdf_path in pdf_files:
        pages = extract_pages_from_pdf(pdf_path)
        chunks = chunker.parse_document(pages, source_filename=pdf_path.name)
        logger.info(f"Extracted {len(chunks)} legal section chunks from '{pdf_path.name}'.")
        all_chunks.extend(chunks)

    # Assign unique chunk IDs
    for idx, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = idx

    logger.info(f"Total legal section chunks across all documents: {len(all_chunks)}")

    if dry_run:
        logger.info("[DRY RUN] Chunking complete. Skipping embedding generation and index creation.")
        summary_file = vectorstore_dir / f"{index_name}_dry_run_chunks.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Dry run chunks preview saved to '{summary_file}'.")
        return

    if not api_key:
        api_key = load_api_key()
    if not api_key:
        logger.error(
            "Embedding cannot proceed without an API Key. "
            "Set GEMINI_API_KEY in your environment or in .env."
        )
        sys.exit(1)

    cache_path = vectorstore_dir / f"{index_name}_embeddings_cache.pkl"
    embeddings = generate_embeddings(
        all_chunks,
        api_key=api_key,
        model=model,
        batch_size=batch_size,
        cache_path=cache_path
    )
    build_and_save_vectorstore(embeddings, all_chunks, vectorstore_dir, index_name=index_name)
    logger.info("Ingestion complete! FAISS vector store is ready for retrieval.")


def main():
    parser = argparse.ArgumentParser(description="GST PDF Legal Ingestion & FAISS Vector Store Pipeline")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory containing raw GST PDFs")
    parser.add_argument("--vectorstore-dir", type=str, default="data/vectorstore", help="Output directory for FAISS index")
    parser.add_argument("--index-name", type=str, default="gst_index", help="Base name for FAISS index and metadata files")
    parser.add_argument("--model", type=str, default="models/gemini-embedding-001", help="Gemini embedding model")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of chunks per embedding API request (default: 20)")
    parser.add_argument("--api-key", type=str, default=None, help="Google Gemini API Key (or set in .env)")
    parser.add_argument("--dry-run", action="store_true", help="Extract and chunk without calling embedding API")

    args = parser.parse_args()

    process_gst_documents(
        raw_dir=Path(args.raw_dir),
        vectorstore_dir=Path(args.vectorstore_dir),
        api_key=args.api_key,
        index_name=args.index_name,
        model=args.model,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
