"""
Background job manager for document ingestion.

Documents are processed asynchronously using Python's ThreadPoolExecutor.
This prevents the upload endpoint from blocking on potentially slow operations
like embedding generation for large documents.

Job lifecycle: PENDING → PROCESSING → COMPLETED | FAILED

In production, replace with Celery + Redis or FastAPI BackgroundTasks with
a persistent queue for durability across restarts.
"""

import uuid
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from threading import Lock

logger = logging.getLogger(__name__)

# In-memory job store (reset on restart — acceptable for this assessment)
_job_store: dict[str, "Job"] = {}
_store_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=4)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    filename: str
    file_path: str
    document_id: str
    status: JobStatus = JobStatus.PENDING
    chunks_created: int = 0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


def create_job(filename: str, file_path: str) -> Job:
    """Create a new ingestion job and return it."""
    job = Job(
        job_id=str(uuid.uuid4()),
        filename=filename,
        file_path=file_path,
        document_id=str(uuid.uuid4()),
    )
    with _store_lock:
        _job_store[job.job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _store_lock:
        return _job_store.get(job_id)


def submit_ingestion_job(job: Job) -> None:
    """Submit an ingestion job to the thread pool for background processing."""
    _executor.submit(_run_ingestion, job)


def _run_ingestion(job: Job) -> None:
    """
    Core ingestion pipeline:
    1. Parse document (PDF or TXT)
    2. Chunk text with sliding window
    3. Embed chunks
    4. Store in FAISS index
    """
    with _store_lock:
        job.status = JobStatus.PROCESSING

    logger.info(f"[Job {job.job_id}] Starting ingestion: {job.filename}")

    try:
        # Step 1: Parse
        from app.services.parser import parse_document
        raw_text = parse_document(job.file_path)
        logger.info(f"[Job {job.job_id}] Parsed {len(raw_text)} characters")

        # Step 2: Chunk
        from app.services.chunker import chunk_text
        chunks = chunk_text(raw_text, source=job.filename)
        if not chunks:
            raise ValueError("No chunks produced — document may be empty or unparseable.")
        logger.info(f"[Job {job.job_id}] Created {len(chunks)} chunks")

        # Step 3 & 4: Embed + Store
        from app.services.vector_store import add_chunks_to_index
        added = add_chunks_to_index(chunks, document_id=job.document_id)

        with _store_lock:
            job.status = JobStatus.COMPLETED
            job.chunks_created = added
            job.completed_at = time.time()

        elapsed = job.completed_at - job.created_at
        logger.info(
            f"[Job {job.job_id}] ✓ Completed in {elapsed:.2f}s | "
            f"{added} chunks indexed for '{job.filename}'"
        )

    except Exception as e:
        logger.error(f"[Job {job.job_id}] ✗ Failed: {e}", exc_info=True)
        with _store_lock:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()
