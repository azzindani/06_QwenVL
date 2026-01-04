"""FastAPI endpoints for document processing with logging."""

import base64
import io
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from .schemas import (
    ExtractionResult,
    OCRResult,
    TableResult,
    FormResult,
    InvoiceResult,
    ContractResult,
    NERResult,
    LayoutResult,
    SpatialResult,
    RecognitionResult,
    DocumentParsingResult,
    BatchJobStatus,
)
from ..tasks import TaskType, get_handler, list_handlers
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter()

# In-memory job storage
_jobs: Dict[str, BatchJobStatus] = {}


def _get_handler(task_type: TaskType):
    """Get task handler with loaded model."""
    from .server import get_model_loader
    
    loader = get_model_loader()
    loaded = loader._loaded_model
    return get_handler(task_type, loaded.model, loaded.processor)


async def _load_image(file: UploadFile) -> Image.Image:
    """Load image from uploaded file."""
    contents = await file.read()
    return Image.open(io.BytesIO(contents)).convert("RGB")


def _image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check requested")
    return {"status": "healthy", "version": "1.0.0"}


@router.get("/tasks")
async def list_available_tasks():
    """List all available task types."""
    logger.debug("Task list requested")
    return {"tasks": [t.value for t in list_handlers()]}


@router.post("/extract/ocr", response_model=OCRResult)
async def extract_ocr(
    file: UploadFile = File(...),
    include_boxes: bool = Form(False),
):
    """Extract text from document using OCR."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] OCR extraction started")
    start_time = time.time()

    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.OCR)
        result = handler.process(image, with_boxes=include_boxes)
        
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] OCR completed in {elapsed:.2f}s")

        return OCRResult(
            success=True,
            text=result.text,
            bounding_boxes=result.bounding_boxes,
            word_count=len(result.text.split()) if result.text else 0,
            visualization=_image_to_base64(result.visualization) if result.visualization else None,
        )
    except Exception as e:
        logger.error(f"[{request_id}] OCR failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/table", response_model=TableResult)
async def extract_table(
    file: UploadFile = File(...),
    output_format: str = Form("json"),
):
    """Extract tables from document."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Table extraction started")
    start_time = time.time()

    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.TABLE)
        result = handler.process(image, output_format=output_format)
        
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Table extraction completed in {elapsed:.2f}s")

        return TableResult(
            success=True,
            tables=result.data.get("tables", []) if result.data else [],
            csv_data=result.data.get("csv") if result.data else None,
            visualization=_image_to_base64(result.visualization) if result.visualization else None,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Table extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/form", response_model=FormResult)
async def extract_form(
    file: UploadFile = File(...),
    extract_signatures: bool = Form(True),
    extract_checkboxes: bool = Form(True),
):
    """Extract form fields from document."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Form extraction started")
    start_time = time.time()

    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.FORM)
        result = handler.process(
            image,
            extract_signatures=extract_signatures,
            extract_checkboxes=extract_checkboxes,
        )
        
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Form extraction completed in {elapsed:.2f}s")

        data = result.data or {}
        return FormResult(
            success=True,
            fields=data.get("fields", []),
            checkboxes=data.get("checkboxes", []),
            signatures=data.get("signatures", []),
            visualization=_image_to_base64(result.visualization) if result.visualization else None,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Form extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/invoice", response_model=InvoiceResult)
async def extract_invoice(
    file: UploadFile = File(...),
    document_type: str = Form("invoice"),
):
    """Parse invoice or receipt."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Invoice extraction started")
    start_time = time.time()

    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.INVOICE)
        result = handler.process(image, document_type=document_type)
        
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Invoice extraction completed in {elapsed:.2f}s")

        data = result.data or {}
        return InvoiceResult(
            success=True,
            header=data.get("header", {}),
            line_items=data.get("line_items", []),
            summary=data.get("summary", {}),
            payment=data.get("payment", {}),
            validation=data.get("validation", {}),
            visualization=_image_to_base64(result.visualization) if result.visualization else None,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Invoice extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/ner", response_model=NERResult)
async def extract_ner(
    file: UploadFile = File(...),
    entity_types: str = Form("all"),
):
    """Extract named entities from document."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] NER extraction started")
    start_time = time.time()

    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.NER)
        types = None if entity_types == "all" else entity_types.split(",")
        result = handler.process(image, entity_types=types)
        
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] NER extraction completed in {elapsed:.2f}s")

        data = result.data or {}
        return NERResult(
            success=True,
            entities=data.get("entities", []),
            entity_counts=result.metadata.get("entity_counts", {}) if result.metadata else {},
            visualization=_image_to_base64(result.visualization) if result.visualization else None,
        )
    except Exception as e:
        logger.error(f"[{request_id}] NER extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/layout", response_model=LayoutResult)
async def analyze_layout(
    file: UploadFile = File(...),
):
    """Analyze document layout."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Layout analysis started")
    start_time = time.time()

    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.LAYOUT)
        result = handler.process(image)
        
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Layout analysis completed in {elapsed:.2f}s")

        data = result.data or {}
        return LayoutResult(
            success=True,
            elements=data.get("elements", []),
            element_count=result.metadata.get("element_count", 0) if result.metadata else 0,
            visualization=_image_to_base64(result.visualization) if result.visualization else None,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Layout analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spatial", response_model=SpatialResult)
async def detect_objects(
    file: UploadFile = File(...),
    object_type: Optional[str] = Form(None),
):
    """Detect objects with spatial understanding."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Spatial detection started")
    start_time = time.time()

    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.SPATIAL)
        
        if object_type:
            result = handler.detect_objects(image, object_type=object_type)
        else:
            result = handler.process(image)
        
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Spatial detection completed in {elapsed:.2f}s")

        return SpatialResult(
            success=True,
            text=result.text,
            visualization=_image_to_base64(result.visualization) if result.visualization else None,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Spatial detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognition", response_model=RecognitionResult)
async def recognize_content(
    file: UploadFile = File(...),
    recognition_type: str = Form("general"),
):
    """Recognize content in image (general, celebrity, food, etc.)."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Recognition started (type={recognition_type})")
    start_time = time.time()

    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.RECOGNITION)
        result = handler.process(image, recognition_type=recognition_type)
        
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Recognition completed in {elapsed:.2f}s")

        return RecognitionResult(
            success=True,
            text=result.text,
            data=result.data,
            visualization=_image_to_base64(result.visualization) if result.visualization else None,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Recognition failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/document", response_model=DocumentParsingResult)
async def parse_document(
    file: UploadFile = File(...),
    output_format: str = Form("html"),
):
    """Parse document structure."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Document parsing started")
    start_time = time.time()

    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.DOCUMENT_PARSING)
        result = handler.process(image, output_format=output_format)
        
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Document parsing completed in {elapsed:.2f}s")

        data = result.data or {}
        return DocumentParsingResult(
            success=True,
            html=result.text,
            bboxes=data.get("bboxes", []),
            visualization=_image_to_base64(result.visualization) if result.visualization else None,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Document parsing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/computer")
async def computer_agent(
    file: UploadFile = File(...),
    instruction: str = Form(...),
):
    """Computer use agent for desktop screenshots."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Computer agent started")
    start_time = time.time()

    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.COMPUTER_AGENT)
        result = handler.process(image, prompt=instruction)
        
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Computer agent completed in {elapsed:.2f}s")

        return {
            "success": True,
            "text": result.text,
            "actions": result.data.get("actions", []) if result.data else [],
            "visualization": _image_to_base64(result.visualization) if result.visualization else None,
        }
    except Exception as e:
        logger.error(f"[{request_id}] Computer agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/mobile")
async def mobile_agent(
    file: UploadFile = File(...),
    instruction: str = Form(...),
):
    """Mobile agent for mobile screenshots."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Mobile agent started")
    start_time = time.time()

    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.MOBILE_AGENT)
        result = handler.process(image, prompt=instruction)
        
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Mobile agent completed in {elapsed:.2f}s")

        return {
            "success": True,
            "text": result.text,
            "actions": result.data.get("actions", []) if result.data else [],
            "visualization": _image_to_base64(result.visualization) if result.visualization else None,
        }
    except Exception as e:
        logger.error(f"[{request_id}] Mobile agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/fields", response_model=ExtractionResult)
async def extract_fields(
    file: UploadFile = File(...),
    schema: str = Form(...),
    preset: Optional[str] = Form(None),
):
    """Extract fields based on schema."""
    import json
    
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Field extraction started")
    start_time = time.time()

    try:
        image = await _load_image(file)
        handler = _get_handler(TaskType.FIELD_EXTRACTION)

        if preset:
            result = handler.process(image, preset=preset)
        else:
            schema_dict = json.loads(schema)
            result = handler.process(image, schema=schema_dict)
        
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Field extraction completed in {elapsed:.2f}s")

        return ExtractionResult(
            success=True,
            text=result.text,
            data=result.data,
            confidence=result.confidence,
        )
    except json.JSONDecodeError:
        logger.error(f"[{request_id}] Invalid JSON schema")
        raise HTTPException(status_code=400, detail="Invalid JSON schema")
    except Exception as e:
        logger.error(f"[{request_id}] Field extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Legacy app for backwards compatibility
app = None


def create_app():
    """Create legacy FastAPI app (deprecated, use server.py instead)."""
    global app
    if app is None:
        from fastapi import FastAPI
        app = FastAPI(
            title="Qwen VL Document Processing API",
            description="Vision-Language model API for document extraction and analysis",
            version="1.0.0",
        )
        app.include_router(router)
    return app
