"""
Production FastAPI Backend for GST Legal Intelligence
------------------------------------------------------
Serves:
  - Startup lifecycle: Loads FAISS index & builds in-memory BM25Okapi keyword index
  - Hybrid Search: Top 15 FAISS + Top 15 BM25, deduplicated
  - Reranking: Cohere Rerank API (top 6 chunks) with fallback
  - Generation: Gemini LLM plain-language generation with specific GST section citations
"""

import os
import re
import time
import pickle
import logging
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Tuple
import uuid

try:
    import faiss
except ModuleNotFoundError:
    raise ImportError(
        "\n\n"
        "===============================================================================\n"
        "[ERROR] 'faiss' is not installed in the current Python environment!\n"
        "You are running the global system Python instead of the project virtual environment (.venv).\n\n"
        "To start the server, run using the project virtual environment:\n"
        "    .venv\\Scripts\\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload\n\n"
        "Or activate the virtual environment in your terminal first:\n"
        "    .venv\\Scripts\\Activate.ps1\n"
        "    uvicorn main:app --host 0.0.0.0 --port 8000 --reload\n"
        "===============================================================================\n"
    ) from None

import numpy as np
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from unstructured.documents.elements import (
    Table as UnstructuredTable,
    ElementMetadata,
    Text as UnstructuredText
)
from unstructured.cleaners.core import clean_extra_whitespace

import warnings
warnings.filterwarnings("ignore")
import google.generativeai as genai
import cohere
from supabase import create_client, Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("gst_chat_backend")

# Load environment configuration
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-south-1"))
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "gst-rag-invoices-slash-020")
S3_INVOICE_BUCKET = S3_BUCKET_NAME  # Backward-compatible alias

# API Security Key Protection for Costly Endpoints (Textract, Cohere, Gemini)
API_KEY_NAME = "X-API-Key"
DEFAULT_API_SECURITY_KEY = "chambers-gst-sec-key-2026"
API_SECURITY_KEY = os.getenv("API_SECURITY_KEY", os.getenv("X_API_KEY", DEFAULT_API_SECURITY_KEY))
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    """
    Guards billing-intensive endpoints (/chat, /validate-bill) against unauthorized requests,
    DDoS abuse, and surprise cloud billing charges from AWS Textract, Google Gemini, and Cohere.
    """
    if not api_key or api_key != API_SECURITY_KEY:
        logger.warning(f"Unauthorized request rejected: missing or invalid '{API_KEY_NAME}' header.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unauthorized: Missing or invalid '{API_KEY_NAME}' header. Costly cloud resources are protected.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    return api_key

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY is missing from environment / .env")

# Initialize Supabase Client
supabase_client: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.warning(f"Failed to initialize Supabase client: {e}")
else:
    logger.warning("SUPABASE_URL or SUPABASE_KEY missing from environment / .env. Supabase chat logging is disabled.")

# Paths (read-only)
BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
FAISS_INDEX_PATH = VECTORSTORE_DIR / "gst_index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "gst_index_metadata.pkl"

EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25 search."""
    return re.findall(r'\w+', text.lower())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and Shutdown lifecycle manager.
    On startup:
      1. Loads existing FAISS index (read-only)
      2. Loads metadata (read-only)
      3. Builds in-memory BM25Okapi keyword index
      4. Stores them in app.state
    """
    logger.info("Starting up FastAPI application...")

    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        logger.error(f"Vector store files missing in '{VECTORSTORE_DIR}'.")
        app.state.faiss_index = None
        app.state.bm25_index = None
        app.state.metadata = None
        app.state.chunks = []
        app.state.is_ready = False
    else:
        # Load FAISS index
        logger.info(f"Loading FAISS index from '{FAISS_INDEX_PATH}'...")
        faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

        # Load metadata
        logger.info(f"Loading metadata from '{METADATA_PATH}'...")
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)

        chunks = metadata.get("chunks", [])
        logger.info(f"Loaded {len(chunks)} text chunks from metadata.")

        # Build in-memory BM25Okapi index
        logger.info("Building in-memory BM25Okapi keyword index...")
        tokenized_corpus = [
            tokenize(c.get("content", c.get("raw_text", "")))
            for c in chunks
        ]
        bm25_index = BM25Okapi(tokenized_corpus)
        logger.info("In-memory BM25Okapi index built successfully.")

        # Store in application state
        app.state.faiss_index = faiss_index
        app.state.bm25_index = bm25_index
        app.state.metadata = metadata
        app.state.chunks = chunks
        app.state.is_ready = True

        logger.info(
            f"Application state ready: {faiss_index.ntotal} FAISS vectors, "
            f"{len(chunks)} BM25 documents."
        )

    yield

    logger.info("Shutting down FastAPI application...")


app = FastAPI(
    title="GST RAG Backend API",
    description="Hybrid FAISS + BM25 retrieval with Cohere Reranking and Gemini QA for GST law.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Legal question regarding GST")


class Citation(BaseModel):
    section: str = Field(..., description="GST Section, Rule, or Schedule name")
    title: str = Field(..., description="Legal title of provision")
    chapter: str = Field(..., description="Chapter context")
    source: str = Field(..., description="Source PDF filename")
    pages: List[int] = Field(..., description="Page numbers in document")


class TopChunk(BaseModel):
    chunk_id: int
    section: str
    title: str
    source: str
    pages: List[int]
    snippet: str


class ChatResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    top_chunks: List[TopChunk]
    rerank_engine: str


class LineItem(BaseModel):
    item_description: Optional[str] = None
    hsn_sac: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    taxable_amount: Optional[float] = None
    cgst_rate: Optional[str] = None
    cgst_amount: Optional[float] = None
    sgst_rate: Optional[str] = None
    sgst_amount: Optional[float] = None
    igst_rate: Optional[str] = None
    igst_amount: Optional[float] = None
    total_tax_rate: Optional[str] = None
    total_amount: Optional[float] = None


class ExtractedTable(BaseModel):
    table_index: int
    rows_count: int
    columns_count: int
    headers: List[str] = []
    rows: List[List[str]] = []
    html: Optional[str] = None
    clean_tsv: Optional[str] = None


class TaxRateSummary(BaseModel):
    detected_tax_rates: List[str] = Field(..., description="Unique tax rates identified across line items and tables (e.g. ['5%', '18%'])")
    cgst_rates: List[str] = []
    sgst_rates: List[str] = []
    igst_rates: List[str] = []


class ValidateBillResponse(BaseModel):
    filename: str
    s3_bucket: str
    s3_key: str
    status: str
    tax_rates: TaxRateSummary
    line_items: List[LineItem]
    tables: List[ExtractedTable]
    formatted_document_text: str


# ---------------------------------------------------------------------------
# Hybrid Search & Reranking Logic
# ---------------------------------------------------------------------------
def run_hybrid_search(query: str, app: FastAPI) -> List[Document]:
    """
    Executes hybrid retrieval:
      - Top 10 from in-memory FAISS index (dense vector search)
      - Top 10 from in-memory BM25 index (sparse keyword search)
      - Deduplicates combined results into a unique LangChain Document list
    """
    faiss_index = app.state.faiss_index
    bm25_index = app.state.bm25_index
    chunks = app.state.chunks

    # 1. FAISS Dense Retrieval (Top 15)
    faiss_docs: List[Document] = []
    try:
        res = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        q_vec = np.array([res["embedding"]], dtype=np.float32)
        faiss.normalize_L2(q_vec)

        scores, faiss_indices = faiss_index.search(q_vec, 15)
        for rank, (score, idx) in enumerate(zip(scores[0], faiss_indices[0])):
            if 0 <= idx < len(chunks):
                c = chunks[idx]
                faiss_docs.append(Document(
                    page_content=c.get("content", c.get("raw_text", "")),
                    metadata={
                        "chunk_id": c.get("chunk_id", idx),
                        "section": c.get("section", ""),
                        "title": c.get("title", ""),
                        "chapter": c.get("chapter", ""),
                        "source": c.get("source", ""),
                        "pages": c.get("pages", []),
                        "raw_text": c.get("raw_text", ""),
                        "faiss_rank": rank,
                        "faiss_score": float(score)
                    }
                ))
    except Exception as e:
        logger.warning(
            f"FAISS dense retrieval unavailable ({type(e).__name__}: {e}). "
            "Proceeding with BM25 sparse keyword retrieval."
        )

    # 2. BM25 Sparse Retrieval (Top 15, or Top 30 if FAISS unavailable)
    top_k_bm25 = 15 if faiss_docs else 30
    query_tokens = tokenize(query)
    bm25_scores = bm25_index.get_scores(query_tokens)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k_bm25]

    bm25_docs: List[Document] = []
    for rank, idx in enumerate(top_bm25_indices):
        if 0 <= idx < len(chunks) and bm25_scores[idx] > 0:
            c = chunks[idx]
            bm25_docs.append(Document(
                page_content=c.get("content", c.get("raw_text", "")),
                metadata={
                    "chunk_id": c.get("chunk_id", idx),
                    "section": c.get("section", ""),
                    "title": c.get("title", ""),
                    "chapter": c.get("chapter", ""),
                    "source": c.get("source", ""),
                    "pages": c.get("pages", []),
                    "raw_text": c.get("raw_text", ""),
                    "bm25_rank": rank,
                    "bm25_score": float(bm25_scores[idx])
                }
            ))

    # 3. Deduplicate combined results
    combined_docs: List[Document] = []
    seen_chunk_ids = set()

    # Interleave FAISS and BM25 to preserve balanced rank priority
    max_len = max(len(faiss_docs), len(bm25_docs))
    for i in range(max_len):
        if i < len(faiss_docs):
            doc = faiss_docs[i]
            cid = doc.metadata.get("chunk_id")
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                combined_docs.append(doc)
        if i < len(bm25_docs):
            doc = bm25_docs[i]
            cid = doc.metadata.get("chunk_id")
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                combined_docs.append(doc)

    logger.info(
        f"Hybrid retrieval fetched {len(faiss_docs)} FAISS + {len(bm25_docs)} BM25 results. "
        f"Deduplicated to {len(combined_docs)} unique candidate chunks."
    )
    return combined_docs


def rerank_top_chunks(
    query: str,
    candidate_docs: List[Document],
    top_n: int = 6
) -> tuple[List[Document], str]:
    """
    Reranks candidate chunks using the Cohere Rerank API.
    Falls back to Reciprocal Rank Fusion (RRF) if COHERE_API_KEY is not configured.
    """
    cohere_key = os.getenv("COHERE_API_KEY")

    if cohere_key:
        try:
            logger.info("Invoking Cohere Rerank API (model: rerank-v3.5)...")
            co = cohere.Client(api_key=cohere_key)
            doc_texts = [d.page_content for d in candidate_docs]

            rerank_result = co.rerank(
                model="rerank-v3.5",
                query=query,
                documents=doc_texts,
                top_n=top_n
            )

            reranked_docs = [candidate_docs[r.index] for r in rerank_result.results]
            logger.info(f"Cohere Rerank successfully selected top {len(reranked_docs)} chunks.")
            return reranked_docs, "Cohere Rerank (rerank-v3.5)"

        except Exception as e:
            logger.warning(f"Cohere Rerank call failed: {e}. Falling back to Reciprocal Rank Fusion.")

    # Fallback: Reciprocal Rank Fusion (RRF)
    logger.info("Applying Reciprocal Rank Fusion fallback for candidate reranking.")
    k = 60
    scored_candidates = []
    for doc in candidate_docs:
        rrf_score = 0.0
        if "faiss_rank" in doc.metadata:
            rrf_score += 1.0 / (k + doc.metadata["faiss_rank"] + 1)
        if "bm25_rank" in doc.metadata:
            rrf_score += 1.0 / (k + doc.metadata["bm25_rank"] + 1)
        scored_candidates.append((rrf_score, doc))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored_candidates[:top_n]]
    engine_label = "Reciprocal Rank Fusion (Set COHERE_API_KEY in .env for Cohere Rerank)"
    return top_docs, engine_label


# ---------------------------------------------------------------------------
# Supabase Logging Utility
# ---------------------------------------------------------------------------
async def log_chat_to_supabase(
    query: str,
    response: str,
    timestamp: Optional[str] = None
) -> None:
    """
    Asynchronously logs incoming /chat queries, generated Gemini responses,
    and timestamps into a Supabase Postgres table named 'chat_logs'.
    Handles any Supabase insertion errors gracefully so they don't crash the API.
    """
    if not supabase_client:
        logger.debug("Supabase client not initialized; skipping chat_logs insertion.")
        return

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    record = {
        "query": query,
        "response": response,
        "timestamp": timestamp,
    }

    try:
        # Offload synchronous PostgREST call to thread pool to keep the event loop non-blocking
        await asyncio.to_thread(
            lambda: supabase_client.table("chat_logs").insert(record).execute()
        )
        logger.info("Successfully logged chat interaction to Supabase table 'chat_logs'.")
    except Exception as e:
        logger.warning(
            f"Supabase insertion error into 'chat_logs': {e}. "
            "Handled gracefully; request will proceed normally."
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    """Root endpoint returning service status and readiness."""
    is_ready = getattr(app.state, "is_ready", False)
    faiss_vectors = app.state.faiss_index.ntotal if is_ready and app.state.faiss_index else 0
    total_chunks = len(app.state.chunks) if is_ready else 0

    return {
        "message": "GST RAG Backend API is running.",
        "status": "online",
        "is_ready": is_ready,
        "faiss_vectors": faiss_vectors,
        "bm25_documents": total_chunks,
        "docs": "/docs"
    }


@app.get("/health")
def health():
    """Health check endpoint reflecting index readiness."""
    is_ready = getattr(app.state, "is_ready", False)
    return {
        "status": "healthy" if is_ready else "degraded",
        "faiss_ready": bool(getattr(app.state, "faiss_index", None) is not None),
        "bm25_ready": bool(getattr(app.state, "bm25_index", None) is not None),
        "total_chunks": len(getattr(app.state, "chunks", []))
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"], dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    """
    POST /chat Endpoint
    1. Runs hybrid search fetching top 10 from FAISS and top 10 from BM25.
    2. Deduplicates results into unique candidate chunks.
    3. Sends combined chunks to Cohere Rerank API to extract top 4 most relevant chunks.
    4. Passes the 4 chunks to Gemini LLM to generate a plain-language answer citing specific GST sections.
    5. Asynchronously logs query, response, and timestamp to Supabase table 'chat_logs'.
    """
    if not getattr(app.state, "is_ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store and BM25 index are not initialized."
        )

    # 1 & 2. Hybrid search & deduplication
    candidate_docs = run_hybrid_search(request.query, app)
    if not candidate_docs:
        no_res_answer = "No relevant GST statutory provisions or rules were found matching your query."
        log_timestamp = datetime.now(timezone.utc).isoformat()
        await log_chat_to_supabase(
            query=request.query,
            response=no_res_answer,
            timestamp=log_timestamp
        )
        return ChatResponse(
            query=request.query,
            answer=no_res_answer,
            citations=[],
            top_chunks=[],
            rerank_engine="none"
        )

    # 3. Cohere Rerank to extract top 6 chunks
    top_6_docs, rerank_engine = rerank_top_chunks(request.query, candidate_docs, top_n=6)

    # 4. Prepare Context & Citations
    context_blocks = []
    citations: List[Citation] = []
    top_chunks_info: List[TopChunk] = []

    for i, doc in enumerate(top_6_docs):
        meta = doc.metadata
        context_blocks.append(
            f"[Provision {i+1}]\n"
            f"Document: {meta.get('source')} (Page(s): {meta.get('pages')})\n"
            f"Chapter: {meta.get('chapter')}\n"
            f"Section: {meta.get('section')} - {meta.get('title')}\n"
            f"Statutory Text:\n{doc.page_content}\n"
        )
        citations.append(Citation(
            section=meta.get("section", ""),
            title=meta.get("title", ""),
            chapter=meta.get("chapter", ""),
            source=meta.get("source", ""),
            pages=meta.get("pages", [])
        ))
        raw = meta.get("raw_text", doc.page_content).strip()
        top_chunks_info.append(TopChunk(
            chunk_id=meta.get("chunk_id", i),
            section=meta.get("section", ""),
            title=meta.get("title", ""),
            source=meta.get("source", ""),
            pages=meta.get("pages", []),
            snippet=raw[:300] + ("..." if len(raw) > 300 else "")
        ))

    context_str = "\n".join(context_blocks)

    # 5. Gemini LLM Plain-Language Generation with Section Citations
    system_prompt = (
        "You are an expert Indian Goods and Services Tax (GST) legal advisor. "
        "Your task is to explain the legal position clearly in plain language so that business owners and tax professionals can understand it.\n\n"
        "Instructions:\n"
        "1. Strictly ground your answer in the 4 provided GST statutory provisions.\n"
        "2. Explicitly cite the specific GST Section(s), Rule(s), and Schedule(s) (e.g., 'Section 17(5) of the CGST Act', 'Rule 42', 'Schedule I') that support each part of your explanation.\n"
        "3. Explain legal conditions and nuances in plain language with bullet points where helpful.\n"
        "4. If the provided context does not address an aspect of the query, clearly state what the law says according to the retrieved text and note what is not specified.\n"
        "5. Be direct, authoritative, and helpful."
    )

    user_prompt = (
        f"STATUTORY GST CONTEXT:\n{context_str}\n\n"
        f"USER QUESTION: {request.query}\n\n"
        f"Please provide a plain-language answer citing specific GST sections:"
    )

    answer_text = None
    last_err = None

    for model_name in GENERATION_MODELS:
        for attempt in range(2):
            try:
                logger.info(f"Generating GST legal answer with model '{model_name}' (attempt {attempt + 1})...")
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_prompt
                )
                response = model.generate_content(user_prompt)
                if response and response.text:
                    answer_text = response.text.strip()
                    break
            except Exception as e:
                last_err = e
                err_str = str(e)
                if "ResourceExhausted" in err_str or "429" in err_str:
                    logger.warning(
                        f"Model '{model_name}' hit rate limit (attempt {attempt + 1}). Retrying in 2s..."
                    )
                    time.sleep(2)
                    continue
                else:
                    logger.warning(f"Model '{model_name}' failed with {e}. Trying fallback model...")
                    break
        if answer_text:
            break

    # If generation failed across models with system_instruction, try combined prompt
    if not answer_text:
        for model_name in GENERATION_MODELS:
            try:
                fallback_prompt = f"{system_prompt}\n\n{user_prompt}"
                model = genai.GenerativeModel(model_name=model_name)
                response = model.generate_content(fallback_prompt)
                if response and response.text:
                    answer_text = response.text.strip()
                    break
            except Exception as e:
                last_err = e
                continue

    if not answer_text:
        logger.error(f"Gemini LLM generation failed across all models: {last_err}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to generate answer from Gemini LLM: {str(last_err)}"
        )

    # Log incoming query, generated Gemini response, and timestamp to Supabase table 'chat_logs'
    log_timestamp = datetime.now(timezone.utc).isoformat()
    await log_chat_to_supabase(
        query=request.query,
        response=answer_text,
        timestamp=log_timestamp
    )

    return ChatResponse(
        query=request.query,
        answer=answer_text,
        citations=citations,
        top_chunks=top_chunks_info,
        rerank_engine=rerank_engine
    )


# ---------------------------------------------------------------------------
# Bill Validation / AWS Textract & Unstructured Extraction Helpers
# ---------------------------------------------------------------------------
def parse_safe_float(val: Any) -> Optional[float]:
    """Extracts float from a numeric string, handling currency symbols and commas."""
    if val is None:
        return None
    cleaned = re.sub(r'[^\d.-]', '', str(val).strip())
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def parse_textract_tables_and_lines(
    blocks: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Reconstructs tables, line items, and raw text lines from AWS Textract blocks.
    """
    block_map = {b["Id"]: b for b in blocks if "Id" in b}
    tables: List[Dict[str, Any]] = []
    line_items: List[Dict[str, Any]] = []
    raw_lines: List[str] = []

    for b in blocks:
        if b.get("BlockType") == "LINE" and b.get("Text"):
            raw_lines.append(b["Text"].strip())

    table_blocks = [b for b in blocks if b.get("BlockType") == "TABLE"]

    for t_idx, t_block in enumerate(table_blocks):
        cell_ids = []
        for rel in t_block.get("Relationships", []):
            if rel.get("Type") == "CHILD":
                cell_ids.extend(rel.get("Ids", []))

        cells = [block_map[cid] for cid in cell_ids if cid in block_map]
        if not cells:
            continue

        max_row = max(c.get("RowIndex", 1) for c in cells)
        max_col = max(c.get("ColumnIndex", 1) for c in cells)

        grid = [["" for _ in range(max_col)] for _ in range(max_row)]

        for c in cells:
            r = c.get("RowIndex", 1) - 1
            col = c.get("ColumnIndex", 1) - 1
            cell_text_parts = []
            for rel in c.get("Relationships", []):
                if rel.get("Type") == "CHILD":
                    for wid in rel.get("Ids", []):
                        if wid in block_map and "Text" in block_map[wid]:
                            cell_text_parts.append(block_map[wid]["Text"])
            cell_text = " ".join(cell_text_parts).strip()
            if 0 <= r < max_row and 0 <= col < max_col:
                grid[r][col] = cell_text

        headers = grid[0] if len(grid) > 0 else []
        rows = grid[1:] if len(grid) > 1 else []

        # Construct clean HTML representation
        html_rows = []
        if headers:
            html_rows.append("<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>")
        body_rows = ["<tr>" + "".join(f"<td>{val}</td>" for val in r) + "</tr>" for r in rows]
        html_str = f"<table>{''.join(html_rows)}<tbody>{''.join(body_rows)}</tbody></table>"

        tables.append({
            "table_index": t_idx + 1,
            "rows_count": len(grid),
            "columns_count": max_col,
            "headers": headers,
            "rows": rows,
            "html": html_str
        })

        # Match columns to line item attributes
        header_map: Dict[str, int] = {}
        for c_idx, h in enumerate(headers):
            h_lower = h.lower()
            if any(k in h_lower for k in ["desc", "item", "particular", "product", "goods", "service"]):
                header_map["description"] = c_idx
            elif any(k in h_lower for k in ["hsn", "sac"]):
                header_map["hsn_sac"] = c_idx
            elif any(k in h_lower for k in ["qty", "quantity", "nos", "units"]):
                header_map["qty"] = c_idx
            elif any(k in h_lower for k in ["rate", "price", "unit price"]):
                header_map["unit_price"] = c_idx
            elif any(k in h_lower for k in ["taxable", "taxable val", "amount"]):
                header_map["taxable_amount"] = c_idx
            elif "cgst" in h_lower and ("rate" in h_lower or "%" in h_lower):
                header_map["cgst_rate"] = c_idx
            elif "cgst" in h_lower and ("amt" in h_lower or "amount" in h_lower):
                header_map["cgst_amount"] = c_idx
            elif "sgst" in h_lower and ("rate" in h_lower or "%" in h_lower):
                header_map["sgst_rate"] = c_idx
            elif "sgst" in h_lower and ("amt" in h_lower or "amount" in h_lower):
                header_map["sgst_amount"] = c_idx
            elif "igst" in h_lower and ("rate" in h_lower or "%" in h_lower):
                header_map["igst_rate"] = c_idx
            elif "igst" in h_lower and ("amt" in h_lower or "amount" in h_lower):
                header_map["igst_amount"] = c_idx
            elif any(k in h_lower for k in ["total", "net amount"]):
                header_map["total_amount"] = c_idx

        for row in rows:
            if not any(val.strip() for val in row):
                continue
            desc = row[header_map["description"]] if "description" in header_map and header_map["description"] < len(row) else None
            hsn = row[header_map["hsn_sac"]] if "hsn_sac" in header_map and header_map["hsn_sac"] < len(row) else None

            # Skip summary/total lines
            if desc and any(k in desc.lower() for k in ["total", "sub total", "grand total", "round off"]):
                continue

            item = {
                "item_description": desc or (row[0] if len(row) > 0 else None),
                "hsn_sac": hsn,
                "quantity": parse_safe_float(row[header_map["qty"]]) if "qty" in header_map and header_map["qty"] < len(row) else None,
                "unit_price": parse_safe_float(row[header_map["unit_price"]]) if "unit_price" in header_map and header_map["unit_price"] < len(row) else None,
                "taxable_amount": parse_safe_float(row[header_map["taxable_amount"]]) if "taxable_amount" in header_map and header_map["taxable_amount"] < len(row) else None,
                "cgst_rate": row[header_map["cgst_rate"]] if "cgst_rate" in header_map and header_map["cgst_rate"] < len(row) else None,
                "cgst_amount": parse_safe_float(row[header_map["cgst_amount"]]) if "cgst_amount" in header_map and header_map["cgst_amount"] < len(row) else None,
                "sgst_rate": row[header_map["sgst_rate"]] if "sgst_rate" in header_map and header_map["sgst_rate"] < len(row) else None,
                "sgst_amount": parse_safe_float(row[header_map["sgst_amount"]]) if "sgst_amount" in header_map and header_map["sgst_amount"] < len(row) else None,
                "igst_rate": row[header_map["igst_rate"]] if "igst_rate" in header_map and header_map["igst_rate"] < len(row) else None,
                "igst_amount": parse_safe_float(row[header_map["igst_amount"]]) if "igst_amount" in header_map and header_map["igst_amount"] < len(row) else None,
                "total_amount": parse_safe_float(row[header_map["total_amount"]]) if "total_amount" in header_map and header_map["total_amount"] < len(row) else None,
            }
            line_items.append(item)

    return tables, line_items, raw_lines


def format_textract_with_unstructured(
    tables: List[Dict[str, Any]],
    raw_lines: List[str]
) -> Tuple[str, List[ExtractedTable]]:
    """
    Uses the unstructured library to format Textract's table grids and text lines cleanly.
    """
    unstructured_elements = []
    formatted_tables: List[ExtractedTable] = []

    for t in tables:
        grid = [t["headers"]] + t["rows"]
        cleaned_grid = [[clean_extra_whitespace(c) for c in row] for row in grid]
        tsv_text = "\n".join(["\t".join(row) for row in cleaned_grid if any(row)])

        table_elem = UnstructuredTable(
            text=tsv_text,
            metadata=ElementMetadata(text_as_html=t["html"])
        )
        unstructured_elements.append(table_elem)
        formatted_tables.append(ExtractedTable(
            table_index=t["table_index"],
            rows_count=len(cleaned_grid),
            columns_count=t["columns_count"],
            headers=cleaned_grid[0] if cleaned_grid else [],
            rows=cleaned_grid[1:] if len(cleaned_grid) > 1 else [],
            html=t["html"],
            clean_tsv=tsv_text
        ))

    cleaned_lines = [clean_extra_whitespace(line) for line in raw_lines if line.strip()]
    for line in cleaned_lines:
        unstructured_elements.append(UnstructuredText(text=line))

    formatted_text = "\n\n".join([str(elem) for elem in unstructured_elements])
    return formatted_text, formatted_tables


def extract_tax_rates_summary(
    tables: List[Dict[str, Any]],
    line_items: List[Dict[str, Any]],
    raw_text: str
) -> TaxRateSummary:
    """
    Extracts all GST tax rates (CGST, SGST, IGST, and composite GST rates)
    from extracted invoice tables, line items, and raw text.
    """
    gst_percent_regex = re.compile(r'\b(0|0\.1|0\.25|1\.5|3|5|6|9|12|14|18|28)(?:\.0+)?\s*%', re.IGNORECASE)
    cgst_regex = re.compile(r'(?:CGST|Central\s+GST)\s*[@:]?\s*(\d+(?:\.\d+)?)\s*%', re.IGNORECASE)
    sgst_regex = re.compile(r'(?:SGST|UTGST|State\s+GST)\s*[@:]?\s*(\d+(?:\.\d+)?)\s*%', re.IGNORECASE)
    igst_regex = re.compile(r'(?:IGST|Integrated\s+GST)\s*[@:]?\s*(\d+(?:\.\d+)?)\s*%', re.IGNORECASE)

    found_rates = set(gst_percent_regex.findall(raw_text))
    cgst_rates = set(cgst_regex.findall(raw_text))
    sgst_rates = set(sgst_regex.findall(raw_text))
    igst_rates = set(igst_regex.findall(raw_text))

    for item in line_items:
        for key, target_set in [
            ("cgst_rate", cgst_rates),
            ("sgst_rate", sgst_rates),
            ("igst_rate", igst_rates),
            ("total_tax_rate", found_rates)
        ]:
            val = str(item.get(key) or "")
            m = re.findall(r'(\d+(?:\.\d+)?)\s*%', val)
            target_set.update(m)

    def normalize(rate_set):
        return sorted(list(set(f"{float(r):g}%" for r in rate_set if r)), key=lambda x: float(x.rstrip('%')))

    norm_cgst = normalize(cgst_rates)
    norm_sgst = normalize(sgst_rates)
    norm_igst = normalize(igst_rates)
    all_norm = set(normalize(found_rates))

    # Pairwise inference (CGST 9% + SGST 9% -> 18% total GST)
    if "9%" in norm_cgst and "9%" in norm_sgst:
        all_norm.add("18%")
    if "6%" in norm_cgst and "6%" in norm_sgst:
        all_norm.add("12%")
    if "2.5%" in norm_cgst and "2.5%" in norm_sgst:
        all_norm.add("5%")
    if "14%" in norm_cgst and "14%" in norm_sgst:
        all_norm.add("28%")
    all_norm.update(norm_igst)

    sorted_detected_rates = sorted(list(all_norm), key=lambda x: float(x.rstrip('%'))) if all_norm else []

    return TaxRateSummary(
        detected_tax_rates=sorted_detected_rates,
        cgst_rates=norm_cgst,
        sgst_rates=norm_sgst,
        igst_rates=norm_igst
    )


# ---------------------------------------------------------------------------
# POST /validate-bill Endpoint
# ---------------------------------------------------------------------------
@app.post("/validate-bill", response_model=ValidateBillResponse, tags=["Invoice Validation"], dependencies=[Depends(verify_api_key)])
async def validate_bill(file: UploadFile = File(...)):
    """
    POST /validate-bill Endpoint
    1. Accepts a PDF or image file upload via UploadFile.
    2. Uses boto3 to temporarily upload this file to the dynamically configured S3 bucket.
    3. Triggers AWS Textract on this S3 object to extract tables and line items.
    4. Uses the unstructured library to format Textract's output cleanly.
    5. Returns the extracted tax rates, line items, and tables as a JSON response.
    6. Cleans up the temporary S3 object in a finally block.
    """
    filename = file.filename or "invoice_upload"
    ext = Path(filename).suffix.lower()
    valid_exts = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    if ext not in valid_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported formats: PDF, PNG, JPG, JPEG, TIFF."
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    # Dynamically resolve AWS Region and S3 Bucket Name
    aws_region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-south-1"))
    bucket_name = os.getenv("S3_BUCKET_NAME", "gst-rag-invoices-slash-020")

    # Configure AWS Session
    session_kwargs = {"region_name": aws_region}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        session_kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        session_kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY

    try:
        s3 = boto3.client("s3", **session_kwargs)
        textract = boto3.client("textract", **session_kwargs)
    except Exception as e:
        logger.error(f"Failed to initialize AWS clients: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AWS service initialization failed: {str(e)}"
        )

    s3_key = f"temp_uploads/{uuid.uuid4().hex}_{filename}"
    uploaded = False

    try:
        # 1. Temporarily upload to S3 bucket
        logger.info(f"Uploading file '{filename}' temporarily to s3://{bucket_name}/{s3_key} (region: {aws_region})...")
        await asyncio.to_thread(
            lambda: s3.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=file_bytes,
                ContentType=file.content_type or "application/octet-stream"
            )
        )
        uploaded = True
        logger.info("Uploaded successfully to S3.")

        # 2. Trigger AWS Textract on the S3 object
        logger.info(f"Triggering AWS Textract on s3://{bucket_name}/{s3_key}...")
        blocks = []
        try:
            res = await asyncio.to_thread(
                lambda: textract.analyze_document(
                    Document={"S3Object": {"Bucket": bucket_name, "Name": s3_key}},
                    FeatureTypes=["TABLES", "FORMS"]
                )
            )
            blocks = res.get("Blocks", [])
        except ClientError as ce:
            err_code = ce.response.get("Error", {}).get("Code", "")
            err_msg = ce.response.get("Error", {}).get("Message", str(ce))
            if "UnsupportedDocumentException" in err_code or "multi-page" in err_msg.lower():
                logger.info("Multi-page PDF detected; triggering async start_document_analysis...")
                job_res = await asyncio.to_thread(
                    lambda: textract.start_document_analysis(
                        DocumentLocation={"S3Object": {"Bucket": bucket_name, "Name": s3_key}},
                        FeatureTypes=["TABLES", "FORMS"]
                    )
                )
                job_id = job_res["JobId"]
                while True:
                    await asyncio.sleep(2)
                    poll_res = await asyncio.to_thread(
                        lambda: textract.get_document_analysis(JobId=job_id)
                    )
                    status_val = poll_res.get("JobStatus")
                    if status_val == "SUCCEEDED":
                        blocks = poll_res.get("Blocks", [])
                        next_tok = poll_res.get("NextToken")
                        while next_tok:
                            more_res = await asyncio.to_thread(
                                lambda: textract.get_document_analysis(JobId=job_id, NextToken=next_tok)
                            )
                            blocks.extend(more_res.get("Blocks", []))
                            next_tok = more_res.get("NextToken")
                        break
                    elif status_val == "FAILED":
                        raise RuntimeError(f"Textract analysis job failed: {poll_res.get('StatusMessage')}")
            else:
                raise

        # 3. Extract tables, line items, and raw lines
        raw_tables, raw_line_items, raw_lines = parse_textract_tables_and_lines(blocks)

        # 4. Format cleanly using the unstructured library
        formatted_text, structured_tables = format_textract_with_unstructured(raw_tables, raw_lines)

        # 5. Extract GST tax rates summary
        full_text_corpus = formatted_text + "\n" + "\n".join(raw_lines)
        tax_summary = extract_tax_rates_summary(raw_tables, raw_line_items, full_text_corpus)

        structured_line_items = [LineItem(**item) for item in raw_line_items]

        return ValidateBillResponse(
            filename=filename,
            s3_bucket=bucket_name,
            s3_key=s3_key,
            status="success",
            tax_rates=tax_summary,
            line_items=structured_line_items,
            tables=structured_tables,
            formatted_document_text=formatted_text
        )

    except ClientError as ce:
        err_msg = ce.response.get("Error", {}).get("Message", str(ce))
        err_code = ce.response.get("Error", {}).get("Code", "")
        logger.error(f"AWS Error ({err_code}): {err_msg}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AWS Textract/S3 error ({err_code}): {err_msg}"
        )
    except Exception as e:
        logger.error(f"Bill validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process bill validation: {str(e)}"
        )
    finally:
        # Clean up temporary S3 object
        if uploaded:
            try:
                await asyncio.to_thread(
                    lambda: s3.delete_object(Bucket=bucket_name, Key=s3_key)
                )
                logger.info(f"Cleaned up temporary S3 object 's3://{bucket_name}/{s3_key}'.")
            except Exception as del_err:
                logger.warning(f"Could not delete temporary S3 object: {del_err}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

