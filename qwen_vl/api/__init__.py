"""API module for document processing service."""

# Lazy imports to avoid heavy dependencies at import time
def __getattr__(name):
    if name == "app":
        from .endpoints import app
        return app
    elif name == "create_app":
        from .endpoints import create_app
        return create_app
    elif name in ("BatchProcessor", "get_batch_processor", "BatchJob", "JobStatus"):
        from .batch import BatchProcessor, get_batch_processor, BatchJob, JobStatus
        return locals()[name]
    elif name in ("WebhookManager", "get_webhook_manager", "EventType"):
        from .webhooks import WebhookManager, get_webhook_manager, EventType
        return locals()[name]
    elif name in ("StorageBackend", "LocalStorage", "S3Storage", "GCSStorage", "create_storage"):
        from .storage import StorageBackend, LocalStorage, S3Storage, GCSStorage, create_storage
        return locals()[name]
    elif name in ("ExportManager", "get_export_manager", "export_to_json", "export_to_csv", "export_to_excel", "export_to_pdf"):
        from .export import ExportManager, get_export_manager, export_to_json, export_to_csv, export_to_excel, export_to_pdf
        return locals()[name]
    elif name in ("schema_to_pydantic", "generate_extraction_models"):
        from .schemas import schema_to_pydantic, generate_extraction_models
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Endpoints
    "app",
    "create_app",
    # Batch processing
    "BatchProcessor",
    "get_batch_processor",
    "BatchJob",
    "JobStatus",
    # Webhooks
    "WebhookManager",
    "get_webhook_manager",
    "EventType",
    # Storage
    "StorageBackend",
    "LocalStorage",
    "S3Storage",
    "GCSStorage",
    "create_storage",
    # Export
    "ExportManager",
    "get_export_manager",
    "export_to_json",
    "export_to_csv",
    "export_to_excel",
    "export_to_pdf",
    # Schemas
    "schema_to_pydantic",
    "generate_extraction_models",
]
