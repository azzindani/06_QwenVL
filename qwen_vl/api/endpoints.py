"""FastAPI application for document processing."""

import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from PIL import Image
import io

from .schemas import (
    ExtractionResult,
    OCRResult,
    TableResult,
    FormResult,
    InvoiceResult,
    ContractResult,
    NERResult,
    BatchJobStatus,
)
from ..tasks import TaskType, get_handler, list_handlers
from ..core.model_loader import ModelLoader


app = FastAPI(
    title="Qwen VL Document Processing API",
    description="Vision-Language model API for document extraction and analysis",
    version="1.0.0",
)

# In-memory job storage (replace with Redis/DB in production)
_jobs: Dict[str, BatchJobStatus] = {}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/tasks")
async def list_available_tasks():
    """List all available task types."""
    return {"tasks": [t.value for t in list_handlers()]}


@app.post("/extract/ocr", response_model=OCRResult)
async def extract_ocr(
    file: UploadFile = File(...),
    include_boxes: bool = Form(False),
):
    """
    Extract text from document using OCR.

    Args:
        file: Image file to process
        include_boxes: Whether to include bounding boxes
    """
    start_time = time.time()

    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.OCR)

        result = handler.process(image, include_boxes=include_boxes)

        return OCRResult(
            success=True,
            text=result.text,
            bounding_boxes=result.bounding_boxes,
            word_count=len(result.text.split()) if result.text else 0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/table", response_model=TableResult)
async def extract_table(
    file: UploadFile = File(...),
    output_format: str = Form("json"),
):
    """
    Extract tables from document.

    Args:
        file: Image file to process
        output_format: Output format (json, csv)
    """
    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.TABLE)

        result = handler.process(image, output_format=output_format)

        return TableResult(
            success=True,
            tables=result.data.get("tables", []) if result.data else [],
            csv_data=result.data.get("csv") if result.data else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/form", response_model=FormResult)
async def extract_form(
    file: UploadFile = File(...),
    extract_signatures: bool = Form(True),
    extract_checkboxes: bool = Form(True),
):
    """
    Extract form fields from document.

    Args:
        file: Image file to process
        extract_signatures: Whether to detect signatures
        extract_checkboxes: Whether to detect checkboxes
    """
    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.FORM)

        result = handler.process(
            image,
            extract_signatures=extract_signatures,
            extract_checkboxes=extract_checkboxes,
        )

        data = result.data or {}
        return FormResult(
            success=True,
            fields=data.get("fields", []),
            checkboxes=data.get("checkboxes", []),
            signatures=data.get("signatures", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/invoice", response_model=InvoiceResult)
async def extract_invoice(
    file: UploadFile = File(...),
    document_type: str = Form("invoice"),
):
    """
    Parse invoice or receipt.

    Args:
        file: Image file to process
        document_type: Type of document (invoice, receipt)
    """
    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.INVOICE)

        result = handler.process(image, document_type=document_type)

        data = result.data or {}
        return InvoiceResult(
            success=True,
            header=data.get("header", {}),
            line_items=data.get("line_items", []),
            summary=data.get("summary", {}),
            payment=data.get("payment", {}),
            validation=data.get("validation", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/contract", response_model=ContractResult)
async def extract_contract(
    file: UploadFile = File(...),
    extract_clauses: bool = Form(True),
    extract_obligations: bool = Form(True),
):
    """
    Analyze contract document.

    Args:
        file: Image file to process
        extract_clauses: Whether to extract clauses
        extract_obligations: Whether to extract obligations
    """
    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.CONTRACT)

        result = handler.process(
            image,
            extract_clauses=extract_clauses,
            extract_obligations=extract_obligations,
        )

        data = result.data or {}
        return ContractResult(
            success=True,
            parties=data.get("parties", []),
            dates=data.get("dates", {}),
            clauses=data.get("clauses", []),
            obligations=data.get("obligations", []),
            key_terms=data.get("key_terms", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/ner", response_model=NERResult)
async def extract_ner(
    file: UploadFile = File(...),
    entity_types: str = Form("all"),
):
    """
    Extract named entities from document.

    Args:
        file: Image file to process
        entity_types: Comma-separated entity types or 'all'
    """
    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.NER)

        # Parse entity types
        types = None if entity_types == "all" else entity_types.split(",")

        result = handler.process(image, entity_types=types)

        data = result.data or {}
        return NERResult(
            success=True,
            entities=data.get("entities", []),
            entity_counts=result.metadata.get("entity_counts", {}) if result.metadata else {},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/fields", response_model=ExtractionResult)
async def extract_fields(
    file: UploadFile = File(...),
    schema: str = Form(...),
    preset: Optional[str] = Form(None),
):
    """
    Extract fields based on schema.

    Args:
        file: Image file to process
        schema: JSON schema string or 'preset:name'
        preset: Preset name (invoice, receipt, id_card, business_card)
    """
    import json

    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.FIELD_EXTRACTION)

        if preset:
            result = handler.process(image, preset=preset)
        else:
            schema_dict = json.loads(schema)
            result = handler.process(image, schema=schema_dict)

        return ExtractionResult(
            success=True,
            text=result.text,
            data=result.data,
            confidence=result.confidence,
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON schema")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _load_image(file: UploadFile) -> Image.Image:
    """Load image from uploaded file."""
    contents = await file.read()
    return Image.open(io.BytesIO(contents)).convert("RGB")


def _get_handler(task_type: TaskType):
    """Get task handler with loaded model."""
    loader = ModelLoader()
    if not loader.is_loaded:
        loader.load()
    return get_handler(task_type, loader.model, loader.processor)


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
