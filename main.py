"""
Main Application Entrypoint
---------------------------
Production FastAPI backend for GST Legal Intelligence RAG system.
Initializes and serves:
  - Dense semantic FAISS vector index
  - Sparse BM25Okapi keyword index (built in-memory on startup)
Stored in application state for high-performance hybrid retrieval endpoints.
"""

import re
import pickle
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any

import faiss
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rank_bm25 import BM25Okapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("gst_backend")

# Paths to existing data files (read-only)
BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
FAISS_INDEX_PATH = VECTORSTORE_DIR / "gst_index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "gst_index_metadata.pkl"


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words for BM25 indexing."""
    return re.findall(r'\w+', text.lower())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup & Shutdown lifecycle manager.
    On startup:
      1. Loads existing FAISS index from data/vectorstore/gst_index.faiss (read-only).
      2. Loads metadata from data/vectorstore/gst_index_metadata.pkl (read-only).
      3. Builds an in-memory BM25Okapi keyword index from the text chunks.
      4. Attaches both indexes, metadata, and chunks to app.state.
    """
    logger.info("Starting up FastAPI application...")

    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        logger.error(
            f"Required index files not found in '{VECTORSTORE_DIR}'. "
            f"Expected '{FAISS_INDEX_PATH.name}' and '{METADATA_PATH.name}'."
        )
        app.state.faiss_index = None
        app.state.bm25_index = None
        app.state.metadata = None
        app.state.chunks = []
        app.state.is_ready = False
    else:
        # 1. Load FAISS index (read-only)
        logger.info(f"Loading FAISS index from '{FAISS_INDEX_PATH}'...")
        faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

        # 2. Load metadata (read-only)
        logger.info(f"Loading metadata from '{METADATA_PATH}'...")
        with open(METADATA_PATH, "rb") as f:
            metadata: Dict[str, Any] = pickle.load(f)

        chunks = metadata.get("chunks", [])
        logger.info(f"Loaded {len(chunks)} text chunks from metadata.")

        # 3. Build in-memory BM25Okapi keyword index from text chunks
        logger.info("Building in-memory BM25Okapi keyword index from text chunks...")
        tokenized_corpus = [
            tokenize(c.get("content", c.get("raw_text", "")))
            for c in chunks
        ]
        bm25_index = BM25Okapi(tokenized_corpus)
        logger.info("BM25Okapi keyword index successfully built in memory.")

        # 4. Store both indexes and metadata in application state
        app.state.faiss_index = faiss_index
        app.state.bm25_index = bm25_index
        app.state.metadata = metadata
        app.state.chunks = chunks
        app.state.is_ready = True

        logger.info(
            f"Application state ready: {faiss_index.ntotal} FAISS vectors (dim={faiss_index.d}), "
            f"BM25 index with {len(chunks)} documents."
        )

    yield

    logger.info("Shutting down FastAPI application...")


# FastAPI application instance with lifespan startup event
app = FastAPI(
    title="GST RAG Backend API",
    description="FastAPI backend serving dense FAISS and sparse BM25 retrieval for GST laws.",
    version="1.0.0",
    lifespan=lifespan
)

# Cross-Origin Resource Sharing (CORS) middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Root endpoint returning service info and index readiness."""
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
