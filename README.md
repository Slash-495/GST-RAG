<div align="center">

# ⚖️ Chambers & Infrastructure
### *GST Legal Intelligence & Automated Invoice Audit*

<p align="center">
  <b>A high-precision Statutory Legal Copilot & Automated Invoice Tax Rate Validator for the Indian Goods and Services Tax framework.</b>
</p>

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS Textract & S3](https://img.shields.io/badge/AWS-Textract%20%7C%20S3-FF9900.svg?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![Cohere Rerank](https://img.shields.io/badge/Cohere-Rerank%20v3.5-39594C.svg?logo=cohere&logoColor=white)](https://cohere.com/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini%20Flash-8E75C2.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![Supabase](https://img.shields.io/badge/Supabase-Audit%20Ledger-3ECF8E.svg?logo=supabase&logoColor=white)](https://supabase.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

<!-- HERO BANNER PLACEHOLDER -->
<img src="https://raw.githubusercontent.com/placeholder/chambers-and-infrastructure/main/docs/images/hero_banner.png" alt="Chambers & Infrastructure Hero Banner" width="100%" onerror="this.src='https://via.placeholder.com/1200x400/271c19/FAF7F2?text=%E2%9A%96%EF%B8%8F+Chambers+%26+Infrastructure+%E2%80%94+GST+Statutory+Counsel+%26+Invoice+Audit';">

<br/><br/>

[Key Features](#-key-features) •
[System Architecture](#-system-architecture) •
[Tech Stack](#-tech-stack) •
[Local Quickstart (Docker & Python)](#-local-quickstart) •
[API Specification](#-api-specification) •
[Environment Configuration](#-environment-configuration)

---

</div>

<br/>

## 📖 Overview

**Chambers & Infrastructure** is an enterprise-grade Legal Artificial Intelligence system engineered to demystify the complexities of India's Goods and Services Tax (CGST, IGST, UTGST, and SGST Acts & Rules).

It pairs **Hybrid Dense-Sparse RAG** with **Computer Vision Document Intelligence** to solve two major enterprise tax compliance hurdles:
1. **Statutory Interpretability**: Answering complex legal tax queries in plain language with exact section, sub-section, rule, chapter, and page-level citations from official legislative gazettes.
2. **Automated Bill Audit**: Extracting, itemizing, and auditing invoice line items, HSN/SAC codes, and applied tax percentages (CGST, SGST, IGST) from scanned invoices and multi-page PDFs using AWS Textract.

---

## 🖼️ User Interface

<div align="center">
  <table>
    <tr>
      <td width="50%" align="center">
        <b>Tab 1: Statutory Legal Copilot (Q&A)</b><br/><br/>
        <!-- COPILOT SCREENSHOT PLACEHOLDER -->
        <img src="docs/images/copilot_ui.png" alt="Statutory Legal Copilot" width="100%" onerror="this.src='https://via.placeholder.com/600x380/FAF7F2/140D0B?text=Tab+1%3A+Legal+Copilot+UI+Screenshot';">
      </td>
      <td width="50%" align="center">
        <b>Tab 2: Invoice & Tax Rate Validator</b><br/><br/>
        <!-- VALIDATOR SCREENSHOT PLACEHOLDER -->
        <img src="docs/images/validator_ui.png" alt="Invoice Validator Dashboard" width="100%" onerror="this.src='https://via.placeholder.com/600x380/FAF7F2/140D0B?text=Tab+2%3A+Invoice+Validator+UI+Screenshot';">
      </td>
    </tr>
  </table>
</div>

---

## 🚀 Key Features

### 1. High-Precision Hybrid Statutory RAG
- **Dense Vector Search**: Powered by FAISS (L2 normalized index) indexing statutory chunks generated via Gemini embeddings (`models/gemini-embedding-001`).
- **Sparse Keyword Search**: In-memory `BM25Okapi` index scanning legal terminology, section numbers, and rule definitions.
- **Cohere Rerank v3.5**: Cross-encoder reranker distilling the top candidates into the 4 most contextually authoritative chunks (with automated Reciprocal Rank Fusion fallback).
- **Statutory Citations**: Generates clear, plain-language legal answers backed by exact section numbers, rule clauses, chapters, source PDF gazette names, and page references.

### 2. AWS Textract Invoice & Bill Validation
- **Multi-Format Ingestion**: Ingests supplier invoices across PDF, PNG, JPG, JPEG, and TIFF formats.
- **Deep Table & Form OCR**: Utilizes AWS Textract `TABLES` and `FORMS` feature types, with automated multi-page asynchronous fallback handling.
- **Line-Item Extraction**: Accurately itemizes HSN/SAC codes, descriptions, quantities, unit prices, taxable values, and applied CGST/SGST/IGST rates.
- **HTML & Structured TSV**: Normalizes extracted document tables and cleans extra whitespace using the `unstructured` document processing library.

### 3. Ephemeral S3 Staging Lifecycle
- **Zero Data Leakage**: Uploads incoming invoices to an ephemeral S3 bucket path (`temp_uploads/{uuid}_{filename}`).
- **Guaranteed Cleanup**: AWS S3 objects are systematically purged in an asynchronous `finally` block immediately following analysis, ensuring zero lingering invoice exposure or storage bloat.

### 4. Supabase Audit Ledger
- **Immutable Query Logs**: Asynchronously logs client queries, LLM responses, timestamps, and metadata into a hosted PostgreSQL `chat_logs` table via Supabase for legal auditability.
- **Resilient Non-Blocking Execution**: Database logging exceptions are handled gracefully without degrading user-facing API performance.

### 5. High-Contrast Law Firm Aesthetic
- **Bespoke UI Design**: Custom Streamlit frontend built with a refined, earthy legal palette (deep walnut brown `#271c19`, soft linen parchment `#FAF7F2`, high-contrast charcoal `#111111`, and classic `Merriweather` serif typography).
- **Real-Time Token Streaming**: Simulates natural legal dictation cadence using `st.write_stream`.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Client ["Client Presentation Layer (Streamlit)"]
        ChatUI["Tab 1: Legal Copilot (Q&A)"]
        UploadUI["Tab 2: Invoice Validator"]
    end

    subgraph BackendAPI ["FastAPI Production Gateway (:8000)"]
        ChatEndpoint["POST /chat"]
        BillEndpoint["POST /validate-bill"]
        HealthEndpoint["GET /health"]
    end

    subgraph RAGPipeline ["Statutory Intelligence Pipeline"]
        FAISS["FAISS Dense Vectorstore (L2)"]
        BM25["BM25Okapi Sparse Keyword Index"]
        Cohere["Cohere Rerank v3.5 API"]
        Gemini["Google Gemini Legal QA"]
        SupaDB[("Supabase Postgres (chat_logs)")]
    end

    subgraph VisionPipeline ["Document Vision Pipeline"]
        S3Bucket[("AWS S3 Staging Bucket")]
        Textract["AWS Textract (Tables + Forms)"]
        Unstructured["Unstructured Formatting Engine"]
        TaxEngine["GST Tax Summary Parser"]
    end

    ChatUI -->|JSON Query| ChatEndpoint
    ChatEndpoint --> FAISS & BM25
    FAISS & BM25 -->|Candidate Chunks| Cohere
    Cohere -->|Top 4 Chunks| Gemini
    Gemini -->|Statutory QA + Citations| ChatEndpoint
    ChatEndpoint -.->|Async Logging| SupaDB
    ChatEndpoint -->|Streaming Answer| ChatUI

    UploadUI -->|Multipart File Upload| BillEndpoint
    BillEndpoint -->|1. Temp Upload| S3Bucket
    S3Bucket -->|2. Trigger Analysis| Textract
    Textract -->|3. Clean Tables & Lines| Unstructured
    Unstructured -->|4. Rate Categorization| TaxEngine
    BillEndpoint -.->|5. Purge Temp File (finally)| S3Bucket
    BillEndpoint -->|JSON Tax Breakdown| UploadUI
```

### 3-Step Invoice Processing Workflow
1. **Secure Ephemeral Staging**: When an invoice is submitted via Streamlit, FastAPI stages the binary file in AWS S3 (`s3://${S3_BUCKET_NAME}/temp_uploads/{uuid}_{filename}`).
2. **Textract Document Analysis**: FastAPI triggers AWS Textract (`TABLES` and `FORMS` feature types), dynamically switching to asynchronous polling if a multi-page document is detected.
3. **Parsing, Verification & Cleanup**: The raw OCR blocks are normalized via the `unstructured` library to reconstruct tables, isolate line-item tax rates (CGST, SGST, IGST), and return clean JSON to the frontend—while the temporary S3 file is instantly purged.

---

## 🛠️ Tech Stack

| Domain | Technology | Purpose & Implementation |
| :--- | :--- | :--- |
| **Frontend** | [Streamlit](https://streamlit.io/) | Interactive dashboard with custom CSS, real-time response streaming, and document preview |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance asynchronous REST API with Swagger documentation (`/docs`) |
| **Language** | [Python 3.12](https://www.python.org/) | Core language runtime utilizing typed Pydantic V2 schemas and async I/O |
| **Vector Index** | [FAISS](https://github.com/facebookresearch/faiss) | In-memory dense vector index for fast statutory chunk similarity retrieval |
| **Keyword Search** | [Rank-BM25](https://github.com/dorianbrown/rank_bm25) | In-memory sparse BM25Okapi keyword index for precise legislative section lookup |
| **Reranking** | [Cohere Rerank](https://cohere.com/) | State-of-the-art cross-encoder (`rerank-v3.5`) with reciprocal rank fusion fallback |
| **Legal LLM** | [Google Gemini](https://ai.google.dev/) | Generative legal synthesis citing official sections, clauses, and schedules |
| **Vision OCR** | [AWS Textract](https://aws.amazon.com/textract/) | Machine learning table and form extraction for scanned documents and invoices |
| **Cloud Storage** | [AWS S3](https://aws.amazon.com/s3/) | Ephemeral document staging lifecycle with automated object deletion |
| **Doc Processing** | [Unstructured](https://unstructured.io/) | Document normalization, table HTML reconstruction, and whitespace cleaning |
| **Audit Ledger** | [Supabase](https://supabase.com/) | Cloud-hosted PostgreSQL table (`chat_logs`) for persistent, immutable audit logging |
| **Container** | [Docker](https://www.docker.com/) | Production-ready `python:3.12-slim` container with OpenMP and healthcheck probes |

---

## 💻 Local Quickstart

### Option A: Running with Docker (Recommended)

The backend is fully containerized, optimized with layer caching, non-root user permissions (`appuser`), and includes OpenMP libraries for FAISS.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/chambers-and-infrastructure.git
cd chambers-and-infrastructure

# 2. Configure your environment file
cp .env.example .env
# Edit .env with your respective API keys

# 3. Build the production Docker image
docker build -t chambers-infrastructure-api:latest .

# 4. Run the container (Mounting your data directory as read-only)
docker run -d \
  --name chambers-infrastructure-api \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data:ro \
  chambers-infrastructure-api:latest

# 5. Check container logs and health probe
docker logs -f chambers-infrastructure-api
curl http://localhost:8000/health
```

---

### Option B: Local Python Virtual Environment

```bash
# 1. Clone and navigate to project root
git clone https://github.com/your-username/chambers-and-infrastructure.git
cd chambers-and-infrastructure

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the FastAPI Backend (Terminal 1)
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 5. Launch the Streamlit Frontend (Terminal 2)
streamlit run frontend/app.py
```

Access the applications:
- **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI OpenAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🔑 Environment Configuration

Create a `.env` file in the project root with the following parameters:

```ini
# ==============================================================================
# Google Gemini Configuration
# ==============================================================================
GEMINI_API_KEY=your_gemini_api_key_here

# ==============================================================================
# Cohere Rerank Configuration
# ==============================================================================
COHERE_API_KEY=your_cohere_api_key_here

# ==============================================================================
# Supabase Audit Logging Configuration
# ==============================================================================
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your_supabase_service_or_anon_key

# ==============================================================================
# AWS Configuration (Textract & S3)
# ==============================================================================
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=ap-south-1
AWS_DEFAULT_REGION=ap-south-1
S3_BUCKET_NAME=gst-rag-invoices-slash-020
```

---

## 📡 API Specification

### `POST /chat`
Submits a plain-language GST question for hybrid retrieval, reranking, and citation generation.

**Request:**
```json
{
  "query": "What motor vehicles are blocked from claiming input tax credit under Section 17?"
}
```

**Response:**
```json
{
  "query": "What motor vehicles are blocked from claiming input tax credit under Section 17?",
  "answer": "Under Section 17(5)(a) of the CGST Act, input tax credit is blocked for motor vehicles designed for transportation of persons having an approved seating capacity of not more than 13 persons (including the driver)...",
  "citations": [
    {
      "section": "Section 17",
      "title": "Apportionment of credit and blocked credits",
      "chapter": "Chapter V: Input Tax Credit",
      "source": "CGST_Act_2017.pdf",
      "pages": [41, 42]
    }
  ],
  "top_chunks": [...],
  "rerank_engine": "Cohere-Rerank-v3.5"
}
```

---

### `POST /validate-bill`
Uploads a tax invoice (PDF or image) for S3 staging, AWS Textract parsing, and tax rate extraction.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file=@sample_invoice.pdf`

**Response:**
```json
{
  "filename": "sample_invoice.pdf",
  "s3_bucket": "gst-rag-invoices-slash-020",
  "s3_key": "temp_uploads/ad9f4a059efd4c8f9a9c9ab2a747ad10_sample_invoice.pdf",
  "status": "success",
  "tax_rates": {
    "detected_tax_rates": ["9%", "18%"],
    "cgst_rates": ["9%"],
    "sgst_rates": ["9%"],
    "igst_rates": []
  },
  "line_items": [
    {
      "item_description": "Legal Consulting Services",
      "hsn_sac": "998311",
      "quantity": 1.0,
      "unit_price": 10000.0,
      "taxable_amount": 10000.0,
      "cgst_rate": "9%",
      "cgst_amount": 900.0,
      "sgst_rate": "9%",
      "sgst_amount": 900.0,
      "total_tax_rate": "18%",
      "total_amount": 11800.0
    }
  ],
  "tables": [...],
  "formatted_document_text": "TAX INVOICE\nGSTIN: 27AABCU9603R1ZM\n..."
}
```

---

## 🛡️ Security & Compliance

- **No Data Retention on S3**: Uploaded invoices are immediately purged in an automated `finally` block to protect client financial confidentiality.
- **Read-Only Vector Store**: FAISS indices and metadata pickles are mounted in read-only mode (`:ro`) to safeguard the statutory embeddings against tampering.
- **Principle of Least Privilege**: Docker containers run under an unprivileged `appuser` (UID 1000) rather than `root`.
- **Statutory Traceability**: Every generated legal answer includes direct citations from the official Indian GST Acts and Rules.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<div align="center">
  <sub>Built with precision for tax professionals, chartered accountants, and compliance officers across India.</sub>
</div>
