"""
GST Vector Store Retrieval & Query Tool
---------------------------------------
Performs semantic search across indexed GST Acts, Rules, and Notifications
using FAISS and Gemini embeddings.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
import faiss

# Import google.generativeai with quiet warnings
import warnings
warnings.filterwarnings("ignore")
import google.generativeai as genai


def search_gst(
    query_text: str,
    top_k: int = 5,
    vectorstore_dir: str = "data/vectorstore",
    index_name: str = "gst_index",
    model: str = "models/gemini-embedding-001"
):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY or GOOGLE_API_KEY not found in .env or environment.")
        sys.exit(1)

    vdir = Path(vectorstore_dir)
    index_file = vdir / f"{index_name}.faiss"
    meta_file = vdir / f"{index_name}_metadata.pkl"

    if not index_file.exists() or not meta_file.exists():
        print(f"[ERROR] Index or metadata not found in '{vdir}'. Run ingest.py first.")
        sys.exit(1)

    # Load FAISS index & metadata
    index = faiss.read_index(str(index_file))
    with open(meta_file, "rb") as f:
        metadata = pickle.load(f)

    # Embed query
    genai.configure(api_key=api_key)
    res = genai.embed_content(
        model=model,
        content=query_text,
        task_type="retrieval_query"
    )
    q_vec = np.array([res["embedding"]], dtype=np.float32)
    faiss.normalize_L2(q_vec)

    scores, indices = index.search(q_vec, top_k)

    print(f"\nQuery: '{query_text}'")
    print("=" * 80)
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        chunk = metadata["chunks"][idx]
        print(f"\n[Rank {rank + 1}] Similarity Score: {score:.4f}")
        print(f"Document : {chunk['source']} (Page(s): {chunk['pages']})")
        print(f"Chapter  : {chunk['chapter']}")
        print(f"Section  : {chunk['section']} - {chunk['title']}")
        print("-" * 80)
        snippet = chunk["raw_text"].strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        print(snippet)
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Query the GST FAISS Vector Store")
    parser.add_argument("query", type=str, nargs="?", default=None, help="The query or question to search")
    parser.add_argument("-k", "--top-k", type=int, default=3, help="Number of results to retrieve (default: 3)")
    parser.add_argument("--vectorstore-dir", type=str, default="data/vectorstore", help="Vector store directory")
    parser.add_argument("--index-name", type=str, default="gst_index", help="Base name of index")

    args = parser.parse_args()

    if args.query:
        search_gst(args.query, top_k=args.top_k, vectorstore_dir=args.vectorstore_dir, index_name=args.index_name)
    else:
        # Interactive loop
        print("\n=== GST Semantic Search (Type 'exit' to quit) ===")
        while True:
            try:
                q = input("\nEnter search query: ").strip()
                if not q or q.lower() in ("exit", "quit"):
                    break
                search_gst(q, top_k=args.top_k, vectorstore_dir=args.vectorstore_dir, index_name=args.index_name)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
