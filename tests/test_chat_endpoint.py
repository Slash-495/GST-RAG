"""
Verification test for POST /chat endpoint in main.py:
- Hybrid search (Top 10 FAISS + Top 10 BM25)
- Deduplication
- Reranking to Top 4
- Gemini LLM plain-language answer generation with GST section citations
"""

import sys
import asyncio
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main

async def test_chat():
    async with main.lifespan(main.app):
        # 1. Test hybrid search directly
        query = "What motor vehicles are blocked from claiming input tax credit under Section 17?"
        print(f"\n--- Testing Hybrid Search for: '{query}' ---")
        candidates = main.run_hybrid_search(query, main.app)
        print(f"Candidate chunks after hybrid search and deduplication: {len(candidates)}")
        assert len(candidates) > 0, "No candidates retrieved from hybrid search"

        # 2. Test Reranking
        print("\n--- Testing Rerank to Top 4 ---")
        top_4, engine = main.rerank_top_chunks(query, candidates, top_n=4)
        print(f"Rerank engine used: {engine}")
        print(f"Top 4 chunks selected: {len(top_4)}")
        for i, doc in enumerate(top_4):
            meta = doc.metadata
            print(f"  [{i+1}] {meta.get('section')} - {meta.get('title')} ({meta.get('source')}, p.{meta.get('pages')})")

        assert len(top_4) <= 4, "Reranking should return at most 4 chunks"

        # 3. Test Full POST /chat Endpoint (Async)
        print("\n--- Testing Full POST /chat Endpoint ---")
        req = main.ChatRequest(query=query)
        response = await main.chat(req)

        print("\n=== Chat Response ===")
        print(f"Query: {response.query}\n")
        print("Answer:\n", response.answer)
        print("\nCitations:")
        for c in response.citations:
            print(f"  - {c.section}: {c.title} ({c.source}, pages: {c.pages})")
        print(f"\nRerank Engine: {response.rerank_engine}")

        # Assertions
        assert response.answer and len(response.answer) > 50, "Answer too short"
        assert len(response.citations) > 0, "Missing citations"
        assert any("17" in c.section or "credit" in c.title.lower() for c in response.citations), "Section 17 missing from citations"

        # 4. Test Supabase error handling resilience
        print("\n--- Testing Supabase Logging & Exception Handling ---")
        # Test 4a: Default/uninitialized state
        await main.log_chat_to_supabase("Test query", "Test response")
        print("Test 4a passed: Graceful no-op when client is uninitialized.")

        # Test 4b: Simulated failing client that raises an exception
        class FailingMockTable:
            def insert(self, record):
                raise RuntimeError("Simulated database connection failure")

        class FailingMockClient:
            def table(self, table_name):
                return FailingMockTable()

        original_client = main.supabase_client
        try:
            main.supabase_client = FailingMockClient()
            # Must not raise exception
            await main.log_chat_to_supabase("Test query", "Test response")
            print("Test 4b passed: Supabase insertion error handled gracefully without raising.")
        finally:
            main.supabase_client = original_client

        print("\nAll POST /chat and Supabase tests passed successfully!")

if __name__ == "__main__":
    asyncio.run(test_chat())
