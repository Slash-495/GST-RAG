"""
Test verification for main.py startup events:
- FAISS index loading
- Metadata loading
- In-memory BM25Okapi index building
- Application state attachment
"""

import sys
import asyncio
import re
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main

async def test_startup():
    async with main.lifespan(main.app):
        # 1. State assertions
        assert main.app.state.faiss_index is not None, "FAISS index missing in app.state"
        assert main.app.state.bm25_index is not None, "BM25 index missing in app.state"
        assert main.app.state.metadata is not None, "Metadata missing in app.state"
        assert len(main.app.state.chunks) == 918, f"Expected 918 chunks, got {len(main.app.state.chunks)}"
        assert main.app.state.is_ready is True, "app.state.is_ready is not True"

        print("\n=== Application State Verification ===")
        print(f"FAISS Total Vectors : {main.app.state.faiss_index.ntotal}")
        print(f"FAISS Dimension     : {main.app.state.faiss_index.d}")
        print(f"BM25 Corpus Size    : {len(main.app.state.chunks)} chunks")

        # 2. Test BM25 keyword retrieval
        query = "motor vehicles input tax credit blocked"
        tokenized_query = re.findall(r'\w+', query.lower())
        scores = main.app.state.bm25_index.get_scores(tokenized_query)
        top_idx = int(scores.argmax())
        best_chunk = main.app.state.chunks[top_idx]

        print(f"\n=== BM25 Keyword Search Test ===")
        print(f"Query      : '{query}'")
        print(f"Top Score  : {scores[top_idx]:.4f}")
        print(f"Document   : {best_chunk['source']}")
        print(f"Provision  : {best_chunk['section']} - {best_chunk['title']}")
        print(f"Page(s)    : {best_chunk['pages']}")
        print(f"Snippet    : {best_chunk['raw_text'][:200].strip()}...")

        # 3. Test Endpoints
        print("\n=== Endpoint Outputs ===")
        print("Root Endpoint   :", main.root())
        print("Health Endpoint :", main.health())

        print("\nAll startup and in-memory index tests passed successfully!")

if __name__ == "__main__":
    asyncio.run(test_startup())
