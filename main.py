"""
Main Application Entrypoint
---------------------------
Standard FastAPI backend application instance.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="GST RAG Backend API",
    description="Backend API for GST Legal Intelligence system.",
    version="1.0.0"
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Root endpoint returning API status."""
    return {
        "message": "GST RAG Backend API is running.",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
