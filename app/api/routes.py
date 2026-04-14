"""
API route definitions.

Endpoints:
  POST /api/v1/upload         — Upload a document for ingestion
  GET  /api/v1/jobs/{job_id}  — Poll ingestion job status
  POST /api/v1/query          — Ask a question against indexed documents
  GET  /api/v1/documents      — List all indexed documents

Rate limits are applied per IP via SlowAPI.
"""

import os
import uuid
import time
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.models.schemas import (
    UploadResponse,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    JobStatus,
)
from app.services.job_manager import create_job, get_job, submit_ingestion_job
from app.services.vector_store import search_index
from app.services.llm import generate_answer

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_MB = 20


# ─────────────────────────────────────────────
# POST /upload
# ─────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document.
    Returns a job_id that can be polled for ingestion status.
    """
    # Validate extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Save file to disk
    safe_name = f"{uuid.uuid4()}{suffix}"
    save_path = os.path.join(settings.UPLOAD_DIR, safe_name)

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            # Check file size
            size_mb = len(content) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large ({size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB",
                )
            f.write(content)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File save failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    # Create and dispatch background job
    job = create_job(filename=file.filename, file_path=save_path)
    submit_ingestion_job(job)

    logger.info(f"Upload accepted: '{file.filename}' → job {job.job_id}")

    return UploadResponse(
        job_id=job.job_id,
        filename=file.filename,
        status=JobStatus.PENDING,
        message="Document accepted. Ingestion started in background.",
    )


# ─────────────────────────────────────────────
# GET /jobs/{job_id}
# ─────────────────────────────────────────────

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll the status of a document ingestion job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return JobStatusResponse(
        job_id=job.job_id,
        filename=job.filename,
        status=JobStatus(job.status.value),
        chunks_created=job.chunks_created if job.status.value == "completed" else None,
        error=job.error,
    )


# ─────────────────────────────────────────────
# POST /query
# ─────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
@limiter.limit(settings.RATE_LIMIT_QUERY)
async def query_documents(request: Request, payload: QueryRequest):
    """
    Ask a question against indexed documents.
    Returns an LLM-generated answer with retrieved context and latency metrics.
    """
    t_total_start = time.time()

    # Retrieve relevant chunks
    retrieved = search_index(
        query=payload.question,
        top_k=payload.top_k,
        document_id=payload.document_id,
    )

    if not retrieved:
        raise HTTPException(
            status_code=404,
            detail="No indexed documents found. Please upload documents first.",
        )

    # Generate answer
    llm_result = generate_answer(
        question=payload.question,
        retrieved_chunks=retrieved,
    )

    total_latency_ms = (time.time() - t_total_start) * 1000

    logger.info(
        f"Query: '{payload.question[:60]}...' | "
        f"chunks={len(retrieved)} | "
        f"total_latency={total_latency_ms:.0f}ms"
    )

    return QueryResponse(
        question=payload.question,
        answer=llm_result["answer"],
        retrieved_chunks=[
            RetrievedChunk(
                text=c["text"],
                source=c["source"],
                chunk_index=c["chunk_index"],
                similarity_score=c["similarity_score"],
            )
            for c in retrieved
        ],
        latency_ms=total_latency_ms,
        model_used=llm_result["model"],
    )


# ─────────────────────────────────────────────
# GET /documents (bonus endpoint)
# ─────────────────────────────────────────────

@router.get("/documents")
async def list_documents():
    """List all uploaded document files."""
    upload_path = Path(settings.UPLOAD_DIR)
    files = list(upload_path.glob("*.*"))
    return {
        "count": len(files),
        "files": [f.name for f in files],
    }
