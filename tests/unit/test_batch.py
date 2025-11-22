"""Unit tests for batch processing system."""

import pytest
from datetime import datetime

from qwen_vl.api.batch import (
    BatchProcessor,
    BatchJob,
    BatchItem,
    JobStatus,
    get_batch_processor,
)


@pytest.mark.unit
class TestBatchItem:
    """Tests for BatchItem dataclass."""

    def test_create_item(self):
        """Test creating a batch item."""
        item = BatchItem(
            item_id="item-1",
            file_path="/path/to/file.png",
        )

        assert item.item_id == "item-1"
        assert item.status == JobStatus.PENDING
        assert item.result is None
        assert item.error is None


@pytest.mark.unit
class TestBatchJob:
    """Tests for BatchJob dataclass."""

    def test_create_job(self):
        """Test creating a batch job."""
        items = [
            BatchItem(item_id="1", file_path="/tmp/1.png"),
            BatchItem(item_id="2", file_path="/tmp/2.png"),
        ]

        job = BatchJob(
            job_id="job-123",
            task_type="ocr",
            items=items,
        )

        assert job.job_id == "job-123"
        assert job.total_items == 2
        assert job.status == JobStatus.PENDING

    def test_progress_calculation(self):
        """Test progress percentage calculation."""
        items = [
            BatchItem(item_id="1", file_path="/tmp/1.png", status=JobStatus.COMPLETED),
            BatchItem(item_id="2", file_path="/tmp/2.png", status=JobStatus.COMPLETED),
            BatchItem(item_id="3", file_path="/tmp/3.png", status=JobStatus.PENDING),
            BatchItem(item_id="4", file_path="/tmp/4.png", status=JobStatus.PENDING),
        ]

        job = BatchJob(job_id="job", task_type="ocr", items=items)

        assert job.processed_items == 2
        assert job.progress == 50.0

    def test_failed_items_count(self):
        """Test failed items counting."""
        items = [
            BatchItem(item_id="1", file_path="/tmp/1.png", status=JobStatus.COMPLETED),
            BatchItem(item_id="2", file_path="/tmp/2.png", status=JobStatus.FAILED),
            BatchItem(item_id="3", file_path="/tmp/3.png", status=JobStatus.FAILED),
        ]

        job = BatchJob(job_id="job", task_type="ocr", items=items)

        assert job.failed_items == 2
        assert job.processed_items == 3


@pytest.mark.unit
class TestBatchProcessor:
    """Tests for BatchProcessor."""

    def test_create_job(self):
        """Test creating a batch job."""
        processor = BatchProcessor()

        job = processor.create_job(
            task_type="ocr",
            file_paths=["/tmp/1.png", "/tmp/2.png"],
            options={"include_boxes": True},
        )

        assert job.job_id is not None
        assert job.task_type == "ocr"
        assert job.total_items == 2
        assert job.options == {"include_boxes": True}

    def test_get_job(self):
        """Test retrieving a job by ID."""
        processor = BatchProcessor()

        job = processor.create_job("ocr", ["/tmp/1.png"])
        retrieved = processor.get_job(job.job_id)

        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_nonexistent_job(self):
        """Test getting non-existent job returns None."""
        processor = BatchProcessor()
        assert processor.get_job("nonexistent") is None

    def test_list_jobs(self):
        """Test listing all jobs."""
        processor = BatchProcessor()

        processor.create_job("ocr", ["/tmp/1.png"])
        processor.create_job("table", ["/tmp/2.png"])

        jobs = processor.list_jobs()
        assert len(jobs) == 2

    def test_list_jobs_by_status(self):
        """Test filtering jobs by status."""
        processor = BatchProcessor()

        job1 = processor.create_job("ocr", ["/tmp/1.png"])
        job2 = processor.create_job("table", ["/tmp/2.png"])
        job2.status = JobStatus.COMPLETED

        pending = processor.list_jobs(status=JobStatus.PENDING)
        completed = processor.list_jobs(status=JobStatus.COMPLETED)

        assert len(pending) == 1
        assert len(completed) == 1

    def test_cancel_pending_job(self):
        """Test cancelling a pending job."""
        processor = BatchProcessor()

        job = processor.create_job("ocr", ["/tmp/1.png"])
        assert processor.cancel_job(job.job_id) is True
        assert job.status == JobStatus.CANCELLED

    def test_cancel_processing_job_fails(self):
        """Test that processing jobs cannot be cancelled."""
        processor = BatchProcessor()

        job = processor.create_job("ocr", ["/tmp/1.png"])
        job.status = JobStatus.PROCESSING

        assert processor.cancel_job(job.job_id) is False

    def test_get_job_results(self):
        """Test getting job results."""
        processor = BatchProcessor()

        job = processor.create_job("ocr", ["/tmp/1.png", "/tmp/2.png"])
        job.items[0].status = JobStatus.COMPLETED
        job.items[0].result = {"text": "Hello"}

        results = processor.get_job_results(job.job_id)

        assert len(results) == 2
        assert results[0]["status"] == "completed"
        assert results[0]["result"] == {"text": "Hello"}

    def test_add_callback(self):
        """Test adding completion callback."""
        processor = BatchProcessor()

        callback_called = []

        def callback(job):
            callback_called.append(job.job_id)

        processor.add_callback(callback)
        assert len(processor._callbacks) == 1


@pytest.mark.unit
class TestGlobalProcessor:
    """Tests for global processor instance."""

    def test_get_batch_processor_singleton(self):
        """Test that get_batch_processor returns same instance."""
        # Reset global
        import qwen_vl.api.batch as batch_module
        batch_module._processor = None

        p1 = get_batch_processor()
        p2 = get_batch_processor()

        assert p1 is p2

        # Reset for other tests
        batch_module._processor = None
