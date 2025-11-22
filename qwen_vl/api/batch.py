"""Batch processing system for document processing."""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from PIL import Image


class JobStatus(str, Enum):
    """Batch job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """Single item in a batch job."""
    item_id: str
    file_path: str
    status: JobStatus = JobStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[int] = None


@dataclass
class BatchJob:
    """Batch processing job."""
    job_id: str
    task_type: str
    items: List[BatchItem]
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_items(self) -> int:
        return len(self.items)

    @property
    def processed_items(self) -> int:
        return sum(1 for item in self.items if item.status in [JobStatus.COMPLETED, JobStatus.FAILED])

    @property
    def failed_items(self) -> int:
        return sum(1 for item in self.items if item.status == JobStatus.FAILED)

    @property
    def progress(self) -> float:
        if self.total_items == 0:
            return 0.0
        return self.processed_items / self.total_items * 100


class BatchProcessor:
    """Process batch document processing jobs."""

    def __init__(self, max_workers: int = 4):
        """
        Initialize batch processor.

        Args:
            max_workers: Maximum concurrent workers
        """
        self.max_workers = max_workers
        self._jobs: Dict[str, BatchJob] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._callbacks: List[Callable] = []

    def create_job(
        self,
        task_type: str,
        file_paths: List[str],
        options: Optional[Dict[str, Any]] = None,
    ) -> BatchJob:
        """
        Create a new batch job.

        Args:
            task_type: Type of task to perform
            file_paths: List of file paths to process
            options: Task-specific options

        Returns:
            Created batch job
        """
        job_id = str(uuid.uuid4())

        items = [
            BatchItem(
                item_id=str(uuid.uuid4()),
                file_path=path,
            )
            for path in file_paths
        ]

        job = BatchJob(
            job_id=job_id,
            task_type=task_type,
            items=items,
            options=options or {},
        )

        self._jobs[job_id] = job
        return job

    def create_job_from_folder(
        self,
        task_type: str,
        folder_path: str,
        patterns: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> BatchJob:
        """
        Create a batch job from all files in a folder.

        Args:
            task_type: Type of task to perform
            folder_path: Path to folder containing files
            patterns: File patterns to match (e.g., ["*.png", "*.jpg"])
            options: Task-specific options

        Returns:
            Created batch job
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder_path}")

        patterns = patterns or ["*.png", "*.jpg", "*.jpeg", "*.pdf", "*.tiff"]

        file_paths = []
        for pattern in patterns:
            file_paths.extend([str(p) for p in folder.glob(pattern)])

        if not file_paths:
            raise ValueError(f"No files found matching patterns: {patterns}")

        return self.create_job(task_type, file_paths, options)

    async def process_job(
        self,
        job_id: str,
        handler_factory: Callable,
    ) -> BatchJob:
        """
        Process a batch job.

        Args:
            job_id: Job ID to process
            handler_factory: Factory function to create task handler

        Returns:
            Completed job
        """
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()

        # Process items in parallel
        loop = asyncio.get_event_loop()
        tasks = []

        for item in job.items:
            task = loop.run_in_executor(
                self._executor,
                self._process_item,
                item,
                handler_factory,
                job.options,
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Update job status
        if all(item.status == JobStatus.COMPLETED for item in job.items):
            job.status = JobStatus.COMPLETED
        elif all(item.status == JobStatus.FAILED for item in job.items):
            job.status = JobStatus.FAILED
        else:
            job.status = JobStatus.COMPLETED

        job.completed_at = datetime.utcnow()

        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback(job)
            except Exception:
                pass

        return job

    def _process_item(
        self,
        item: BatchItem,
        handler_factory: Callable,
        options: Dict[str, Any],
    ) -> None:
        """Process a single batch item."""
        import time

        start_time = time.time()
        item.status = JobStatus.PROCESSING

        try:
            # Load image
            image = Image.open(item.file_path).convert("RGB")

            # Get handler and process
            handler = handler_factory()
            result = handler.process(image, **options)

            item.result = {
                "text": result.text,
                "data": result.data,
                "metadata": result.metadata,
            }
            item.status = JobStatus.COMPLETED

        except Exception as e:
            item.error = str(e)
            item.status = JobStatus.FAILED

        finally:
            item.processing_time_ms = int((time.time() - start_time) * 1000)

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
    ) -> List[BatchJob]:
        """
        List all jobs.

        Args:
            status: Filter by status

        Returns:
            List of jobs
        """
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled
        """
        job = self.get_job(job_id)
        if not job:
            return False

        if job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            return True

        return False

    def add_callback(self, callback: Callable[[BatchJob], None]) -> None:
        """Add a completion callback."""
        self._callbacks.append(callback)

    def get_job_results(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Get results for a completed job.

        Args:
            job_id: Job ID

        Returns:
            List of results
        """
        job = self.get_job(job_id)
        if not job:
            return []

        return [
            {
                "item_id": item.item_id,
                "file_path": item.file_path,
                "status": item.status.value,
                "result": item.result,
                "error": item.error,
                "processing_time_ms": item.processing_time_ms,
            }
            for item in job.items
        ]


# Global processor instance
_processor: Optional[BatchProcessor] = None


def get_batch_processor(max_workers: int = 4) -> BatchProcessor:
    """Get or create the global batch processor."""
    global _processor
    if _processor is None:
        _processor = BatchProcessor(max_workers=max_workers)
    return _processor


if __name__ == "__main__":
    print("=" * 60)
    print("BATCH PROCESSOR TEST")
    print("=" * 60)

    processor = BatchProcessor(max_workers=2)

    # Create a test job
    job = processor.create_job(
        task_type="ocr",
        file_paths=["/tmp/test1.png", "/tmp/test2.png"],
        options={"include_boxes": True},
    )

    print(f"  Job ID: {job.job_id}")
    print(f"  Total items: {job.total_items}")
    print(f"  Status: {job.status}")

    print("=" * 60)
