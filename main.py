"""
Production FastAPI Backend for GST Legal Intelligence
------------------------------------------------------
Serves:
  - Startup lifecycle: Loads FAISS index & builds in-memory BM25Okapi keyword index
  - Hybrid Search: Top 10 FAISS + Top 10 BM25, deduplicated
  - Reranking: Cohere Rerank API (top 4 chunks) with fallback
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
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

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

    # 1. FAISS Dense Retrieval (Top 10)
    faiss_docs: List[Document] = []
    try:
        res = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        q_vec = np.array([res["embedding"]], dtype=np.float32)
        faiss.normalize_L2(q_vec)

        scores, faiss_indices = faiss_index.search(q_vec, 10)
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

    # 2. BM25 Sparse Retrieval (Top 10, or Top 20 if FAISS unavailable)
    top_k_bm25 = 10 if faiss_docs else 20
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
    top_n: int = 4
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


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
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

    # 3. Cohere Rerank to extract top 4 chunks
    top_4_docs, rerank_engine = rerank_top_chunks(request.query, candidate_docs, top_n=4)

    # 4. Prepare Context & Citations
    context_blocks = []
    citations: List[Citation] = []
    top_chunks_info: List[TopChunk] = []

    for i, doc in enumerate(top_4_docs):
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
