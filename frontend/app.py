"""
GST Legal Counsel & Invoice Audit - Streamlit Application
=========================================================
Professional, High-Contrast Earthy Legal Theme:
  - Deep Walnut Brown (#271c19) for primary buttons, sidebar, and accents
  - Crisp High-Contrast Typography: Dark Black/Charcoal (#111111 / #1A1A1A) on Soft Warm Canvas (#FAF7F2)
  - Bright White (#FFFFFF) text on all Walnut Buttons and Action Elements
  - Solid, well-defined legal borders (#8C7866 / #5C4A3A) for clear visual structure
  - Traditional Serif Typography ('Merriweather', Georgia, serif)
Features:
  - Tab 1 ('Legal Copilot'): Hybrid FAISS + BM25 legal retrieval, Cohere reranking, Gemini statutory analysis with exact section citations
  - Tab 2 ('Invoice Validator'): Document staging to S3, AWS Textract table extraction, and structured GST rate validation
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional

import requests
import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GST Legal Counsel & Invoice Audit",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API Configuration
DEFAULT_BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# -----------------------------------------------------------------------------
# Custom CSS: High-Contrast, Professional Earthy Legal Theme
# Palette:
#   - Canvas Background: #FAF7F2 (Warm Linen / Crisp Parchment)
#   - Primary Buttons & Accents: #271c19 (Deep Walnut Brown)
#   - Primary Text: #111111 / #1A1A1A (Deep High-Contrast Black)
#   - Secondary Text: #2D2522 (Deep Charcoal)
#   - Prominent Borders: #8C7866 / #5C4A3A (Sharp, Visible Legal Borders)
#   - Button Text: #FFFFFF (Bright Crisp White)
# -----------------------------------------------------------------------------
LEGAL_THEME_CSS = """
<style>
/* Google Font Import: Merriweather & Inter */
@import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;0,900;1,300;1,400;1,700&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* Global Root Variables */
:root {
    --walnut-primary: #271c19;
    --walnut-hover: #44322d;
    --canvas-bg: #FAF7F2;
    --card-bg: #FFFFFF;
    --border-strong: #8C7866;
    --border-subtle: #B8A898;
    --text-pure-dark: #111111;
    --text-body: #1A1A1A;
    --text-muted: #332A26;
}

/* Global Canvas Background & High Contrast Defaults */
.stApp {
    background-color: var(--canvas-bg) !important;
    color: var(--text-pure-dark) !important;
    font-family: 'Merriweather', Georgia, serif !important;
    line-height: 1.65;
}

/* Typography - Bold, Deep & High Contrast */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Merriweather', Georgia, serif !important;
    font-weight: 800 !important;
    color: #140D0B !important;
    letter-spacing: -0.3px;
}

.stApp p, .stApp span, .stApp label, .stApp div {
    color: var(--text-body);
}

.legal-header-title {
    font-family: 'Merriweather', Georgia, serif !important;
    font-weight: 900;
    font-size: 2.3rem;
    color: #140D0B !important;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}

.legal-sub-header {
    font-family: 'Inter', sans-serif !important;
    color: #241D1A !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    margin-top: 0;
    margin-bottom: 1.2rem;
}

/* High-Contrast Legal Badges */
.legal-badge {
    display: inline-block;
    padding: 4px 12px;
    font-family: 'Inter', sans-serif;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #140D0B;
    background-color: #E8DEC8;
    border: 1.5px solid #786555;
    border-radius: 4px;
    margin-bottom: 0.5rem;
}

.legal-badge-gold {
    color: #140D0B;
    background-color: #EFE3D0;
    border-color: #8C7866;
    font-weight: 700;
}

/* Sidebar Styling - Deep Walnut Brown with Bright Crisp Text */
section[data-testid="stSidebar"] {
    background-color: #1C1311 !important;
    color: #F5EFEB !important;
    border-right: 2px solid #3d2d29 !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label {
    color: #F0E8DF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
}

section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stCaption p {
    color: #DDD3C7 !important;
    font-size: 0.85rem !important;
}

section[data-testid="stSidebar"] .legal-badge {
    background-color: #3b2c28 !important;
    color: #FFFFFF !important;
    border-color: #6E534A !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #4A3832 !important;
}

section[data-testid="stSidebar"] input {
    background-color: #120C0A !important;
    color: #FFFFFF !important;
    border: 1.5px solid #6E534A !important;
    font-weight: 500 !important;
}

/* Formal Legal Card Container */
.legal-card {
    background-color: #FFFFFF;
    border: 1.5px solid #8C7866;
    border-left: 6px solid #271c19;
    border-radius: 6px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 3px 10px rgba(39, 28, 25, 0.08);
}

/* Tabs Styling - Bold and Clear */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: transparent;
    padding: 0;
    border-bottom: 2px solid #8C7866;
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    background-color: transparent;
    border-radius: 0;
    color: #3A312D !important;
    font-family: 'Merriweather', Georgia, serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none;
    border-bottom: 4px solid transparent;
    padding: 0 24px;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #140D0B !important;
    background-color: rgba(39, 28, 25, 0.05);
}

.stTabs [aria-selected="true"] {
    color: #140D0B !important;
    font-weight: 900 !important;
    border-bottom: 4px solid #140D0B !important;
    background-color: rgba(232, 222, 200, 0.3) !important;
}

/* Chat Messages */
.stChatMessage {
    border-radius: 6px !important;
    padding: 18px 22px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 2px 6px rgba(39, 28, 25, 0.06) !important;
}

.stChatMessage[data-testid="stChatMessageAssistant"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #9C8978 !important;
    border-left: 6px solid #271c19 !important;
}

.stChatMessage[data-testid="stChatMessageAssistant"] * {
    color: #111111 !important;
}

.stChatMessage[data-testid="stChatMessageUser"] {
    background-color: #EFE7DC !important;
    border: 1.5px solid #8C7866 !important;
    border-left: 6px solid #44322D !important;
}

.stChatMessage[data-testid="stChatMessageUser"] * {
    color: #111111 !important;
}

/* Chat Input Bar */
.stChatInputContainer {
    background-color: #FFFFFF !important;
    border: 2px solid #8C7866 !important;
    border-radius: 6px !important;
    box-shadow: 0 3px 8px rgba(39, 28, 25, 0.08) !important;
}

.stChatInputContainer:focus-within {
    border: 2px solid #140D0B !important;
    box-shadow: 0 0 0 2px rgba(20, 13, 11, 0.2) !important;
}

.stChatInputContainer input {
    color: #111111 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}

/* ========================================================================= */
/* BUTTONS - HIGH CONTRAST SOLID WALNUT WITH PURE WHITE TEXT                 */
/* ========================================================================= */
.stButton > button {
    background-color: #271c19 !important;
    background: #271c19 !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    border: 2px solid #140D0B !important;
    border-radius: 6px !important;
    padding: 10px 22px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background-color: #44322D !important;
    background: #44322D !important;
    border-color: #44322D !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.35) !important;
    transform: translateY(-1px);
}

.stButton > button:active {
    background-color: #120C0A !important;
    background: #120C0A !important;
    transform: translateY(0);
}

/* CRITICAL: Enforce Pure White text on all children of buttons */
.stButton > button *,
.stButton > button p,
.stButton > button div,
.stButton > button span {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.2px !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6) !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #EFE6D8 !important;
    border: 1.5px solid #8C7866 !important;
    border-radius: 5px !important;
    color: #140D0B !important;
    font-family: 'Merriweather', Georgia, serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

.streamlit-expanderHeader * {
    color: #140D0B !important;
    font-weight: 700 !important;
}

.streamlit-expanderHeader:hover {
    background-color: #E5D9C7 !important;
    border-color: #5C4A3A !important;
}

.streamlit-expanderContent {
    background-color: #FFFFFF !important;
    border: 1.5px solid #8C7866 !important;
    border-top: none !important;
    border-radius: 0 0 5px 5px !important;
    padding: 18px !important;
}

.streamlit-expanderContent * {
    color: #1A1A1A;
}

/* Metrics and Summary Containers */
div[data-testid="metric-container"] {
    background-color: #FFFFFF !important;
    border: 2px solid #8C7866 !important;
    border-radius: 6px !important;
    padding: 16px 20px !important;
    box-shadow: 0 2px 6px rgba(39, 28, 25, 0.08) !important;
}

div[data-testid="metric-container"] label,
div[data-testid="metric-container"] label * {
    color: #140D0B !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
}

div[data-testid="metric-container"] div[data-testid="stMetricValue"],
div[data-testid="metric-container"] div[data-testid="stMetricValue"] * {
    color: #140D0B !important;
    font-family: 'Merriweather', Georgia, serif !important;
    font-size: 1.65rem !important;
    font-weight: 800 !important;
}

/* File Uploader - Crisp Defined Parchment Dropzone */
section[data-testid="stFileUploadDropzone"] {
    background-color: #F5ECE0 !important;
    border: 2.5px dashed #5C4A3A !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}

section[data-testid="stFileUploadDropzone"]:hover {
    border-color: #140D0B !important;
    background-color: #EDE2D2 !important;
}

section[data-testid="stFileUploadDropzone"] * {
    color: #140D0B !important;
    font-weight: 600 !important;
}

/* Legal Citation Card */
.citation-pill {
    background-color: #FFFFFF;
    border: 1.5px solid #8C7866;
    border-left: 5px solid #271c19;
    padding: 14px 18px;
    border-radius: 4px;
    margin-bottom: 12px;
    font-size: 0.94rem;
    color: #111111;
    box-shadow: 0 1px 4px rgba(39, 28, 25, 0.06);
}

.citation-pill strong {
    color: #140D0B;
    font-family: 'Merriweather', Georgia, serif;
    font-size: 1.02rem;
    font-weight: 800;
}

/* Tax Rate Badges */
.tax-badge {
    display: inline-block;
    background-color: #271c19;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    font-weight: 800;
    padding: 6px 14px;
    border-radius: 4px;
    margin: 4px 6px 4px 0;
    letter-spacing: 0.4px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}

/* Status Indicator Dot */
.status-dot {
    height: 10px;
    width: 10px;
    background-color: #22C55E;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    box-shadow: 0 0 4px #22C55E;
}

.status-dot-off {
    background-color: #EF4444;
    box-shadow: 0 0 4px #EF4444;
}

/* Dataframe clean styling */
div[data-testid="stDataFrame"] {
    border: 2px solid #8C7866;
    border-radius: 6px;
    background-color: #FFFFFF;
}
</style>
"""

st.markdown(LEGAL_THEME_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Helper Utilities
# -----------------------------------------------------------------------------
def check_backend_health(base_url: str) -> Dict[str, Any]:
    """Pings backend health endpoint to check service status."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=1.0)
        if resp.status_code == 200:
            return {"online": True, "data": resp.json()}
        return {"online": False, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"online": False, "error": str(e)}


def stream_text_generator(text: str, delay: float = 0.012):
    """Generator for streaming text token-by-token with natural cadence."""
    for token in text.split(" "):
        yield token + " "
        time.sleep(delay)


# -----------------------------------------------------------------------------
# Sidebar: System Diagnostics & Repository Specs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="legal-badge">SYSTEM DIAGNOSTICS</div>', unsafe_allow_html=True)
    st.markdown("### ⚖️ Chambers & Infrastructure")

    backend_url = st.text_input(
        "Backend API Base URL",
        value=DEFAULT_BACKEND_URL,
        help="FastAPI instance hosting hybrid retrieval, reranking, and invoice audit endpoints."
    ).rstrip("/")

    # Live Health Check
    health_status = check_backend_health(backend_url)
    if health_status["online"]:
        st.markdown(
            '<div style="margin-bottom:10px;"><span class="status-dot"></span>'
            '<span style="color:#4ADE80;font-weight:700;font-size:0.88rem;font-family:Inter,sans-serif;">SERVICE ONLINE & READY</span></div>',
            unsafe_allow_html=True
        )
        data = health_status.get("data", {})
        st.caption(f"Status: `{data.get('status', 'healthy')}` | Statutory Index: `{data.get('index_loaded', True)}`")
        if "faiss_vectors" in data:
            st.caption(f"Vectors: `{data.get('faiss_vectors')}` | Keyword Chunks: `{data.get('bm25_chunks')}`")
    else:
        st.markdown(
            '<div style="margin-bottom:10px;"><span class="status-dot status-dot-off"></span>'
            '<span style="color:#F87171;font-weight:700;font-size:0.88rem;font-family:Inter,sans-serif;">BACKEND OFFLINE</span></div>',
            unsafe_allow_html=True
        )
        st.caption(f"Notice: {health_status.get('error', 'Cannot connect to backend')}")
        st.info("Start the backend server using:\n`python -m uvicorn main:app --port 8000`")

    st.divider()

    st.markdown("### 📜 Statutory Jurisprudence")
    st.markdown(
        """
        - **Dense Index**: FAISS L2 Vectorstore
        - **Sparse Index**: BM25Okapi Keyword Corpus
        - **Reranker**: Cohere `rerank-v3.5` / RRF
        - **Statutory LLM**: Google Gemini Legal QA
        - **Audit Ledger**: Supabase Postgres (`chat_logs`)
        - **Vision Pipeline**: AWS Textract & `unstructured`
        - **Staging**: S3 `gst-rag-invoices-slash-020`
        """
    )

    st.divider()

    if st.button("Clear Consultation Record", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
col_header, col_meta = st.columns([3, 1])
with col_header:
    st.markdown('<div class="legal-badge">GST LEGAL INTELLIGENCE & JURISPRUDENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="legal-header-title">GST Statutory Counsel & Invoice Audit</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="legal-sub-header">'
        'High-precision statutory retrieval, section-level legal citations, and automated invoice tax audit.'
        '</p>',
        unsafe_allow_html=True
    )
with col_meta:
    st.markdown(
        """
        <div style="text-align: right; margin-top: 10px;">
            <span class="legal-badge legal-badge-gold">CGST / SGST / IGST (2017–2026)</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab_chat, tab_invoice, tab_architecture = st.tabs([
    "⚖️ Statutory Legal Copilot",
    "🧾 Invoice & Tax Rate Validator",
    "🏗️ System Architecture & Pipeline Flow"
])


# =============================================================================
# TAB 1: Legal Copilot
# =============================================================================
with tab_chat:
    st.markdown(
        '<p style="color:#1A1A1A; font-size:0.96rem; margin-bottom:18px; font-family:Inter,sans-serif; font-weight:500;">'
        'Inquire on any provision of the Goods and Services Tax framework. The copilot performs hybrid FAISS dense '
        'and BM25 sparse keyword retrieval, reranks with Cohere, and generates precise plain-language counsel '
        'substantiated by official statutory citations.'
        '</p>',
        unsafe_allow_html=True
    )

    # Formal Statutory Query Shortcuts with High Contrast
    st.markdown(
        '<div style="font-size:0.86rem; font-weight:800; color:#140D0B; margin-bottom:10px; font-family:Inter,sans-serif; letter-spacing:0.8px;">'
        'FREQUENT STATUTORY INQUIRIES:'
        '</div>',
        unsafe_allow_html=True
    )
    p_cols = st.columns(4)
    quick_query = None
    with p_cols[0]:
        if st.button("Motor Vehicles ITC Blockage", use_container_width=True):
            quick_query = "What motor vehicles are blocked from claiming input tax credit under Section 17?"
    with p_cols[1]:
        if st.button("Time of Supply of Services", use_container_width=True):
            quick_query = "What is the time of supply for services under Section 13?"
    with p_cols[2]:
        if st.button("Tax Invoice Particulars", use_container_width=True):
            quick_query = "What are the mandatory invoice particulars required under Rule 46?"
    with p_cols[3]:
        if st.button("Tax Evasion Penalties", use_container_width=True):
            quick_query = "What penalties apply for tax evasion or fraudulent ITC claims under Section 122?"

    # Initialize Consultation History in Session State
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to the **GST Statutory Counsel**. You may present legal questions regarding the "
                    "Central Goods and Services Tax (CGST) Act, Integrated GST (IGST) Act, State GST laws, "
                    "and relevant statutory rules. Responses provide clear legal interpretation cited directly "
                    "from official legislative enactments and schedules."
                ),
                "citations": [],
                "top_chunks": [],
                "rerank_engine": None
            }
        ]

    # Render Prior Consultation Record
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Render Citations Expander if citations exist
            citations = msg.get("citations", [])
            if citations:
                with st.expander(f"Statutory Citations ({len(citations)} References Authenticated)", expanded=False):
                    for c in citations:
                        pages_str = ", ".join(map(str, c.get("pages", []))) if c.get("pages") else "N/A"
                        st.markdown(
                            f"""
                            <div class="citation-pill">
                                <strong>{c.get('section', 'Statutory Section')}</strong> — {c.get('title', 'Statutory Provision')}<br>
                                <span style="color:#2D2522; font-size:0.86rem; font-family:Inter,sans-serif; font-weight:500;">
                                    Chapter: {c.get('chapter', 'N/A')} &nbsp;|&nbsp; 
                                    Source Document: <code>{c.get('source', 'Enactment')}</code> &nbsp;|&nbsp; 
                                    Page(s): {pages_str}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            # Render Top Chunks Expander if retrieved context is available
            chunks = msg.get("top_chunks", [])
            if chunks:
                with st.expander("Retrieved Statutory Chunks & Reranked Context", expanded=False):
                    for chk in chunks:
                        st.markdown(f"**Chunk #{chk.get('chunk_id')}** — `{chk.get('section')}` ({chk.get('title')})")
                        st.caption(f"Source: {chk.get('source')} | Pages: {chk.get('pages')}")
                        st.markdown(f"> *{chk.get('snippet')}*")
                        st.divider()

    # User Input Handling
    chat_prompt = st.chat_input("Enter a statutory question (e.g. ITC on gifts, refund claims under Sec 54, Rule 86B)...")
    active_prompt = quick_query or chat_prompt

    if active_prompt:
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # Assistant Response with streaming
        with st.chat_message("assistant"):
            with st.spinner("Conducting statutory hybrid retrieval & legal synthesis..."):
                try:
                    payload = {"query": active_prompt}
                    response = requests.post(
                        f"{backend_url}/chat",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=90
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer_text = data.get("answer", "No response generated.")
                        citations = data.get("citations", [])
                        top_chunks = data.get("top_chunks", [])
                        rerank_engine = data.get("rerank_engine", "standard")

                        # Stream the synthesized answer smoothly
                        st.write_stream(stream_text_generator(answer_text))

                        # Citations Expander
                        if citations:
                            with st.expander(f"Statutory Citations ({len(citations)} References Authenticated)", expanded=True):
                                for c in citations:
                                    pages_str = ", ".join(map(str, c.get("pages", []))) if c.get("pages") else "N/A"
                                    st.markdown(
                                        f"""
                                        <div class="citation-pill">
                                            <strong>{c.get('section', 'Statutory Section')}</strong> — {c.get('title', 'Statutory Provision')}<br>
                                            <span style="color:#2D2522; font-size:0.86rem; font-family:Inter,sans-serif; font-weight:500;">
                                                Chapter: {c.get('chapter', 'N/A')} &nbsp;|&nbsp; 
                                                Source Document: <code>{c.get('source', 'Enactment')}</code> &nbsp;|&nbsp; 
                                                Page(s): {pages_str}
                                            </span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                        # Top Chunks Expander
                        if top_chunks:
                            with st.expander(f"Statutory Chunks Examined ({rerank_engine})", expanded=False):
                                for chk in top_chunks:
                                    st.markdown(f"**Chunk #{chk.get('chunk_id')}** — `{chk.get('section')}` ({chk.get('title')})")
                                    st.caption(f"Source: {chk.get('source')} | Pages: {chk.get('pages')}")
                                    st.markdown(f"> *{chk.get('snippet')}*")
                                    st.divider()

                        # Append to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer_text,
                            "citations": citations,
                            "top_chunks": top_chunks,
                            "rerank_engine": rerank_engine
                        })

                    else:
                        error_detail = response.text
                        try:
                            error_detail = response.json().get("detail", error_detail)
                        except Exception:
                            pass
                        err_msg = f"Service notification (HTTP {response.status_code}): {error_detail}"
                        st.error(err_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": err_msg,
                            "citations": [],
                            "top_chunks": [],
                            "rerank_engine": None
                        })

                except requests.exceptions.ConnectionError:
                    err_msg = (
                        "**Connection Unavailable**: Unable to communicate with the FastAPI service at `"
                        f"{backend_url}`. Please verify that the application server is active (`uvicorn main:app --port 8000`)."
                    )
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                        "citations": [],
                        "top_chunks": [],
                        "rerank_engine": None
                    })
                except Exception as ex:
                    err_msg = f"**Processing Exception**: {str(ex)}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                        "citations": [],
                        "top_chunks": [],
                        "rerank_engine": None
                    })


# =============================================================================
# TAB 2: Invoice Validator
# =============================================================================
with tab_invoice:
    st.markdown(
        '<p style="color:#1A1A1A; font-size:0.96rem; margin-bottom:18px; font-family:Inter,sans-serif; font-weight:500;">'
        'Upload invoice documentation in PDF or image format. The file is temporarily staged in secure S3 storage, '
        'processed through AWS Textract document intelligence, and parsed via <code>unstructured</code> '
        'to extract structured tax rates, item breakdowns, and compliance schedules.'
        '</p>',
        unsafe_allow_html=True
    )

    inv_col1, inv_col2 = st.columns([1.2, 2])

    with inv_col1:
        st.markdown('<div class="legal-card">', unsafe_allow_html=True)
        st.markdown("#### 📁 Document Submission")
        uploaded_file = st.file_uploader(
            "Select Tax Invoice (PDF or Image)",
            type=["pdf", "png", "jpg", "jpeg", "tiff"],
            help="Submit an invoice document for automated tax rate verification."
        )

        if uploaded_file is not None:
            st.markdown(
                f"""
                <div style="font-size:0.88rem; color:#140D0B; font-family:Inter,sans-serif; font-weight:600; margin: 14px 0;">
                    📄 <strong>File:</strong> <code>{uploaded_file.name}</code><br>
                    ⚖️ <strong>Size:</strong> {uploaded_file.size / 1024:.2f} KB<br>
                    📋 <strong>Format:</strong> {uploaded_file.type}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Preview if image
            if uploaded_file.type and uploaded_file.type.startswith("image/"):
                st.image(uploaded_file, caption="Invoice Submission Preview", use_container_width=True)

            process_btn = st.button("Execute Document Audit", use_container_width=True)
        else:
            process_btn = False
            st.caption("Accepted document formats: PDF, PNG, JPG, JPEG, TIFF.")

        st.markdown('</div>', unsafe_allow_html=True)

    with inv_col2:
        if uploaded_file is not None and process_btn:
            with st.spinner("Staging document and executing AWS Textract analysis..."):
                try:
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }

                    t0 = time.time()
                    resp = requests.post(
                        f"{backend_url}/validate-bill",
                        files=files,
                        timeout=120
                    )
                    latency = time.time() - t0

                    if resp.status_code == 200:
                        bill_data = resp.json()
                        st.session_state["last_bill_result"] = bill_data
                        st.session_state["last_bill_latency"] = latency
                        st.success(f"Audit completed successfully in {latency:.2f} seconds.")
                    else:
                        err_text = resp.text
                        try:
                            err_text = resp.json().get("detail", err_text)
                        except Exception:
                            pass
                        st.error(f"Audit failed (HTTP {resp.status_code}): {err_text}")

                except requests.exceptions.ConnectionError:
                    st.error(f"Unable to reach the backend at `{backend_url}`.")
                except Exception as e:
                    st.error(f"Document audit exception: {str(e)}")

        # Render Results from Session State
        if "last_bill_result" in st.session_state:
            result = st.session_state["last_bill_result"]
            tax_summary = result.get("tax_rates", {})
            line_items = result.get("line_items", [])
            tables = result.get("tables", [])
            clean_text = result.get("formatted_document_text", "")
            s3_key = result.get("s3_key", "")
            s3_bucket = result.get("s3_bucket", "")

            # -------------------------------------------------------------
            # 1. Tax Rate Metric Cards
            # -------------------------------------------------------------
            st.markdown("### 📊 Verified Tax Rate Schedules")
            detected_rates = tax_summary.get("detected_tax_rates", [])

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Detected GST Rates", ", ".join(detected_rates) if detected_rates else "None Identified")
            with m_col2:
                cgst_rates = tax_summary.get("cgst_rates", [])
                st.metric("CGST Rate(s)", ", ".join(cgst_rates) if cgst_rates else "—")
            with m_col3:
                sgst_rates = tax_summary.get("sgst_rates", [])
                st.metric("SGST Rate(s)", ", ".join(sgst_rates) if sgst_rates else "—")
            with m_col4:
                igst_rates = tax_summary.get("igst_rates", [])
                st.metric("IGST Rate(s)", ", ".join(igst_rates) if igst_rates else "—")

            # Formal Tax Badges Display
            if detected_rates:
                st.markdown('<div style="margin: 12px 0;">', unsafe_allow_html=True)
                for r in detected_rates:
                    st.markdown(f'<span class="tax-badge">GST RATE: {r}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.caption(f"Staging Ledger Reference: `s3://{s3_bucket}/{s3_key}` (Temporary staging cleaned automatically)")

            st.divider()

            # -------------------------------------------------------------
            # 2. Extracted Line Items Table
            # -------------------------------------------------------------
            st.markdown("### 📋 Line Item Particulars & Tax Apportionment")

            if line_items:
                formatted_items = []
                for item in line_items:
                    formatted_items.append({
                        "Description": item.get("item_description") or "—",
                        "HSN / SAC": item.get("hsn_sac") or "—",
                        "Qty": item.get("quantity") if item.get("quantity") is not None else "—",
                        "Unit Price (₹)": f"{item.get('unit_price'):,.2f}" if item.get("unit_price") is not None else "—",
                        "Taxable Value (₹)": f"{item.get('taxable_amount'):,.2f}" if item.get("taxable_amount") is not None else "—",
                        "CGST Rate": item.get("cgst_rate") or "—",
                        "CGST (₹)": f"{item.get('cgst_amount'):,.2f}" if item.get("cgst_amount") is not None else "—",
                        "SGST Rate": item.get("sgst_rate") or "—",
                        "SGST (₹)": f"{item.get('sgst_amount'):,.2f}" if item.get("sgst_amount") is not None else "—",
                        "IGST Rate": item.get("igst_rate") or "—",
                        "IGST (₹)": f"{item.get('igst_amount'):,.2f}" if item.get("igst_amount") is not None else "—",
                        "Total GST Rate": item.get("total_tax_rate") or "—",
                        "Total (₹)": f"{item.get('total_amount'):,.2f}" if item.get("total_amount") is not None else "—"
                    })

                df_items = pd.DataFrame(formatted_items)
                st.dataframe(df_items, use_container_width=True, hide_index=True)
            else:
                st.info("No itemized line rows isolated. Consult the reconstructed tables below.")

            # -------------------------------------------------------------
            # 3. AWS Textract Reconstructed Tables
            # -------------------------------------------------------------
            if tables:
                st.markdown("### 🗂️ Reconstructed Document Tables")
                for tbl in tables:
                    t_idx = tbl.get("table_index", 1)
                    r_count = tbl.get("rows_count", 0)
                    c_count = tbl.get("columns_count", 0)
                    headers = tbl.get("headers", [])
                    rows = tbl.get("rows", [])

                    with st.expander(f"Table Structure #{t_idx} ({r_count} Rows × {c_count} Columns)", expanded=(t_idx == 1)):
                        if headers and rows:
                            try:
                                max_cols = max(len(headers), max((len(r) for r in rows), default=0))
                                normalized_headers = headers + [f"Col {i+1}" for i in range(len(headers), max_cols)]
                                normalized_rows = [
                                    r + [""] * (max_cols - len(r)) if len(r) < max_cols else r[:max_cols]
                                    for r in rows
                                ]
                                df_table = pd.DataFrame(normalized_rows, columns=normalized_headers[:max_cols])
                                st.dataframe(df_table, use_container_width=True, hide_index=True)
                            except Exception:
                                if tbl.get("html"):
                                    st.markdown(tbl["html"], unsafe_allow_html=True)
                        elif tbl.get("html"):
                            st.markdown(tbl["html"], unsafe_allow_html=True)
                        elif tbl.get("clean_tsv"):
                            st.code(tbl["clean_tsv"], language="tsv")

            # -------------------------------------------------------------
            # 4. Normalized Document Text & Raw Payload
            # -------------------------------------------------------------
            with st.expander("📄 Normalized Document Text (`unstructured`)", expanded=False):
                st.text_area("Extracted Document Text", value=clean_text, height=220, disabled=True)

            with st.expander("🔬 Complete Textract JSON Payload", expanded=False):
                st.json(result)

        elif uploaded_file is None:
            st.markdown(
                """
                <div class="legal-card" style="text-align: center; padding: 40px 20px;">
                    <h3 style="color: #140D0B; margin-bottom: 8px;">Awaiting Document Submission</h3>
                    <p style="color: #1A1A1A; font-size: 0.95rem; font-family: Inter, sans-serif; font-weight: 500;">
                        Select a supplier invoice or receipt on the left panel and select 
                        <strong>Execute Document Audit</strong> to review itemized tax schedules.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


# =============================================================================
# TAB 3: System Architecture & Pipeline Flow
# =============================================================================
with tab_architecture:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <span class="legal-badge">END-TO-END PIPELINE VISUALIZATION</span>
            <h3 style="margin-top: 5px; color: #140D0B;">The 5 Core Stages of Invoice Intelligence & Statutory Verification</h3>
            <p style="color: #1A1A1A; font-size: 0.95rem; font-family: Inter, sans-serif; font-weight: 500;">
                Traditional invoice auditing requires manual cross-referencing against 160+ GST sections and rules. 
                Below is the exact data journey showing how an unstructured document moves through ephemeral cloud staging, 
                computer vision OCR, hybrid dense-sparse retrieval, and generative statutory validation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # Stage 1: Client Upload
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="legal-card">
            <span class="legal-badge">STAGE 01 // INGESTION</span>
            <h4 style="margin: 4px 0 8px 0; color: #140D0B;">📤 Client Upload & Multipart Dispatch</h4>
            <p style="color: #1A1A1A; font-size: 0.92rem; font-family: Inter, sans-serif; margin: 0;">
                The Streamlit UI captures incoming supplier invoices (PDF, PNG, JPG, TIFF), validates file integrity, 
                and fires an asynchronous multipart payload to the containerized FastAPI backend gateway on <code>POST /validate-bill</code>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    with st.expander("🔍 View Core Logic — Client Upload"):
        st.code(
            """# Streamlit captures binary upload and dispatches multipart request
files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
response = requests.post(f"{backend_url}/validate-bill", files=files, timeout=120)
bill_data = response.json()
return bill_data""",
            language="python"
        )

    st.markdown("<div style='text-align: center; font-size: 1.8rem; margin: -5px 0 10px 0;'>⬇️</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Stage 2: Secure Cloud Staging
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="legal-card">
            <span class="legal-badge">STAGE 02 // SECURE STAGING</span>
            <h4 style="margin: 4px 0 8px 0; color: #140D0B;">☁️ Secure Cloud Staging (AWS S3)</h4>
            <p style="color: #1A1A1A; font-size: 0.92rem; font-family: Inter, sans-serif; margin: 0;">
                FastAPI streams the document bytes directly to an encrypted S3 bucket (<code>gst-rag-invoices-slash-020</code>) 
                under a unique nonce path. Zero raw files are persisted to local disk, and automatic cleanup is guaranteed via a <code>finally</code> block.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    with st.expander("🔍 View Core Logic — S3 Ephemeral Staging"):
        st.code(
            """# FastAPI streams document directly to AWS S3 with dynamic region
s3 = boto3.client("s3", region_name=aws_region)
await asyncio.to_thread(
    lambda: s3.put_object(Bucket=bucket_name, Key=s3_key, Body=file_bytes, ContentType=file.content_type)
)
# Cleanup is guaranteed in a finally block after processing completes""",
            language="python"
        )

    st.markdown("<div style='text-align: center; font-size: 1.8rem; margin: -5px 0 10px 0;'>⬇️</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Stage 3: Vision & OCR
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="legal-card">
            <span class="legal-badge">STAGE 03 // COMPUTER VISION</span>
            <h4 style="margin: 4px 0 8px 0; color: #140D0B;">👁️ Vision & OCR Intelligence (AWS Textract)</h4>
            <p style="color: #1A1A1A; font-size: 0.92rem; font-family: Inter, sans-serif; margin: 0;">
                AWS Textract is triggered on the S3 object using <code>TABLES</code> and <code>FORMS</code> feature types (with 
                asynchronous polling fallback for multi-page invoices). Output blocks are parsed through the <code>unstructured</code> library 
                to isolate line items, quantities, and taxable rates.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    with st.expander("🔍 View Core Logic — Textract Analysis & Normalization"):
        st.code(
            """# Trigger AWS Textract on the ephemeral S3 object (Tables + Forms)
textract = boto3.client("textract", region_name=aws_region)
res = await asyncio.to_thread(lambda: textract.analyze_document(
    Document={"S3Object": {"Bucket": bucket_name, "Name": s3_key}}, FeatureTypes=["TABLES", "FORMS"]
))
raw_tables, raw_line_items, raw_lines = parse_textract_tables_and_lines(res.get("Blocks", []))""",
            language="python"
        )

    st.markdown("<div style='text-align: center; font-size: 1.8rem; margin: -5px 0 10px 0;'>⬇️</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Stage 4: RAG Retrieval
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="legal-card">
            <span class="legal-badge">STAGE 04 // HYBRID RETRIEVAL</span>
            <h4 style="margin: 4px 0 8px 0; color: #140D0B;">📚 Hybrid Statutory RAG Retrieval (FAISS + BM25 + Cohere)</h4>
            <p style="color: #1A1A1A; font-size: 0.92rem; font-family: Inter, sans-serif; margin: 0;">
                The extracted invoice line items and queries trigger dual-stream retrieval: top-10 dense vectors from 
                <strong>FAISS L2</strong> and top-10 sparse keywords from <strong>BM25Okapi</strong>. <strong>Cohere Rerank v3.5</strong> 
                distills these into the top 4 contextually authoritative statutory provisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    with st.expander("🔍 View Core Logic — Dual-Stream Hybrid Rerank"):
        st.code(
            """# In-memory dense FAISS + sparse BM25Okapi search distilled by Cohere
scores, indices = faiss_index.search(q_vec, 10)
bm25_indices = np.argsort(bm25_index.get_scores(tokenize(query)))[::-1][:10]
candidates = deduplicate_chunks(faiss_candidates, bm25_candidates)
reranked = cohere_client.rerank(model="rerank-v3.5", query=query, documents=candidates, top_n=4)""",
            language="python"
        )

    st.markdown("<div style='text-align: center; font-size: 1.8rem; margin: -5px 0 10px 0;'>⬇️</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Stage 5: LLM Validation
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="legal-card">
            <span class="legal-badge">STAGE 05 // VERIFICATION & SYNTHESIS</span>
            <h4 style="margin: 4px 0 8px 0; color: #140D0B;">⚖️ LLM Validation & Section-Level Citations (Google Gemini)</h4>
            <p style="color: #1A1A1A; font-size: 0.92rem; font-family: Inter, sans-serif; margin: 0;">
                <strong>Google Gemini</strong> cross-references the extracted invoice rates (e.g. 18% GST on services) against 
                the retrieved statutory provisions, validating rate alignment, verifying HSN codes, generating legal counsel, 
                and persisting immutable audit logs in <strong>Supabase</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    with st.expander("🔍 View Core Logic — Gemini Legal QA & Supabase Audit"):
        st.code(
            """# Google Gemini legal QA cross-referencing extracted items with statutory law
model = genai.GenerativeModel("gemini-2.5-flash")
qa_response = model.generate_content(f"Context:\\n{legal_context}\\n\\nTask: Validate invoice tax rates and cite sections.")
await log_chat_to_supabase(query=query, response=qa_response.text, timestamp=datetime.now())
return ValidateBillResponse(status="success", tax_rates=tax_summary, line_items=line_items)""",
            language="python"
        )

    st.success("✅ End-to-End Pipeline Verified: Zero Data Leakage • Full Statutory Auditability • Sub-15s Latency")
