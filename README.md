# Qwen3-VL Production Deployment

A production-ready Vision-Language model service built on Qwen3-VL, featuring document processing, OCR, field extraction, and enterprise-grade inference capabilities with a Gradio UI interface.

---

## Model Support

| Model | VRAM (4-bit) | Best For | Variant |
|-------|--------------|----------|---------|
| **Qwen3-VL-2B** | ~4GB | High-throughput simple OCR, basic extraction | Instruct |
| **Qwen3-VL-4B** | ~8GB | Balanced performance, standard documents | Instruct |
| **Qwen3-VL-8B** | ~16GB | Complex documents, multi-step reasoning | Instruct / Thinking |

**Thinking variants**: Better for complex reasoning tasks (validation, cross-field verification, multi-step extraction)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GRADIO UI LAYER                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Chat Panel  │  │ File Upload │  │ Task Selection Panel    │  │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API / SERVICE LAYER                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              InferenceService (Singleton)               │    │
│  │  - Request validation                                   │    │
│  │  - Task routing                                         │    │
│  │  - Response formatting                                  │    │
│  └─────────────────────────┬───────────────────────────────┘    │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CORE INFERENCE LAYER                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            UnifiedQwenInference Engine                  │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                    │
│  ┌─────────────┬───────────┼───────────┬─────────────┐          │
│  ▼             ▼           ▼           ▼             ▼          │
│ ┌───┐       ┌───┐       ┌───┐       ┌───┐       ┌───┐          │
│ │OCR│       │Doc│       │Rec│       │Agt│       │Vid│          │
│ └───┘       └───┘       └───┘       └───┘       └───┘          │
│  Task Handlers (Pluggable)                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL / INFRASTRUCTURE                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Model Loader │  │  Processor   │  │   Hardware   │           │
│  │ (Quantized)  │  │  (Tokenizer) │  │  Detection   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Input (Text + Image/Video)
         │
         ▼
    Gradio UI
         │
         ▼
  InferenceService.infer()
         │
         ▼
  Input Validation & Preprocessing
         │
         ▼
  Task Router → Select Handler
         │
         ▼
  process_vision_info() → Extract embeddings
         │
         ▼
  apply_chat_template() → Format for model
         │
         ▼
  model.generate() → Streaming inference
         │
         ▼
  Post-processing (JSON/HTML parsing)
         │
         ▼
  Visualization (bounding boxes, etc.)
         │
         ▼
  Response to User
```

---

## Features

### Document Processing Capabilities

| Feature | Description | Phase |
|---------|-------------|-------|
| **OCR** | Full-page and line-level text extraction with bounding boxes | 1 |
| **Layout Analysis** | Detect document structure (headers, sections, columns) | 1 |
| **Table Extraction** | Convert tables to structured JSON/CSV | 2 |
| **Field Extraction** | Extract specific fields using schemas | 2 |
| **NER** | Named Entity Recognition (names, dates, amounts, orgs) | 2 |
| **Form Understanding** | Key-value pair extraction from forms | 3 |
| **Invoice/Receipt Parsing** | Structured extraction for financial documents | 3 |
| **Contract Analysis** | Extract clauses, parties, dates, obligations | 3 |

### Developer & Enterprise Features

| Feature | Description | Phase |
|---------|-------------|-------|
| Model quantization (4-bit) | Reduce VRAM usage | 0 |
| Streaming responses | Real-time token generation | 0 |
| Gradio UI | Web interface for all tasks | 1 |
| Schema definition | User-defined extraction templates | 2 |
| API generation | Generate Pydantic models from schemas | 4 |
| Batch processing | Process multiple documents | 4 |
| Webhook integration | Async notifications on completion | 4 |
| Multi-tenant support | Isolated workspaces per user/org | 5 |
| Usage monitoring | Requests, latency, cost tracking | 5 |
| Audit logging | Compliance and traceability | 5 |

---

## Directory Structure

```
qwen_vl/
├── config.py                    # Centralized configuration
├── main.py                      # Application entry point
├── requirements.txt             # Dependencies
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Multi-service orchestration
├── .env.example                 # Environment template
│
├── core/                        # Core business logic
│   ├── __init__.py
│   ├── inference_engine.py      # UnifiedQwenInference class
│   ├── model_loader.py          # Model initialization & caching
│   └── hardware_detection.py    # GPU detection & allocation
│
├── tasks/                       # Task-specific handlers
│   ├── __init__.py
│   ├── base.py                  # BaseTaskHandler abstract class
│   ├── ocr.py                   # OCR task handler
│   ├── document_parsing.py      # Document parsing handler
│   ├── recognition.py           # Image recognition handler
│   ├── spatial.py               # Spatial understanding handler
│   ├── video.py                 # Video understanding handler
│   ├── computer_agent.py        # Computer agent handler
│   └── mobile_agent.py          # Mobile agent handler
│
├── services/                    # Service layer
│   ├── __init__.py
│   └── inference_service.py     # Main inference service
│
├── ui/                          # User interface
│   ├── __init__.py
│   ├── gradio_app.py            # Main Gradio application
│   ├── components/              # Reusable UI components
│   │   ├── __init__.py
│   │   ├── chat_panel.py        # Chat interface
│   │   ├── file_upload.py       # File handling
│   │   └── task_selector.py     # Task selection
│   └── styles/                  # CSS and themes
│       └── theme.py
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── visualization.py         # Bounding box drawing, etc.
│   ├── parsers.py               # JSON/HTML/XML parsing
│   ├── image_processing.py      # Image resize, format conversion
│   ├── video_processing.py      # Video frame extraction
│   └── logger.py                # Logging configuration
│
├── schemas/                     # Data models
│   ├── __init__.py
│   ├── requests.py              # Input validation models
│   └── responses.py             # Output models
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── unit/                    # Unit tests (no GPU)
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_parsers.py
│   │   ├── test_visualization.py
│   │   ├── test_image_processing.py
│   │   ├── test_schemas.py
│   │   └── test_task_handlers.py
│   └── integration/             # Integration tests (requires GPU)
│       ├── __init__.py
│       ├── test_model_loader.py
│       ├── test_inference_engine.py
│       ├── test_tasks.py
│       └── test_gradio_app.py
│
├── notebooks/                   # Original notebooks (reference)
│   └── *.ipynb
│
└── docs/                        # Documentation
    ├── API.md                   # API reference
    ├── DEPLOYMENT.md            # Deployment guide
    └── CONFIGURATION.md         # Config reference
```

---

## Expectations

### Performance Requirements

| Metric | Target | Notes |
|--------|--------|-------|
| **Cold start** | < 60s | Model loading with quantization |
| **Inference latency** | < 3s (2B), < 5s (4B), < 8s (8B) | Single image, 512 tokens |
| **Throughput** | 15 req/min (2B), 10 req/min (4B), 6 req/min (8B) | Single GPU |
| **Memory usage** | < 4GB VRAM | 2B model with 4-bit quantization |
| **Memory usage** | < 8GB VRAM | 4B model with 4-bit quantization |
| **Memory usage** | < 16GB VRAM | 8B model with 4-bit quantization |

### Reliability Requirements

- **Uptime**: 99% availability during business hours
- **Error rate**: < 1% failed requests
- **Graceful degradation**: Return meaningful errors, never crash
- **Recovery**: Auto-restart on failure

### Security Requirements

- Input validation on all user inputs
- File type validation for uploads
- Size limits on uploaded files (max 20MB images, 100MB videos)
- No execution of user-provided code
- Secure temporary file handling with cleanup

---

## Roadmap

### Phase 0 - Foundation

**Goal**: Project infrastructure and model loading

#### Tasks
- [ ] Initialize directory structure
- [ ] Create config.py with model/inference settings
- [ ] Set up logging infrastructure (JSON format)
- [ ] Create requirements.txt with all dependencies
- [ ] Implement hardware_detection.py (GPU detection, VRAM check)
- [ ] Implement model_loader.py with singleton pattern
- [ ] Support Qwen3-VL 2B/4B/8B model selection
- [ ] 4-bit quantization configuration

#### Tests
| Test | Type | Description |
|------|------|-------------|
| `test_config.py` | Unit | Load config, env var override, validation |
| `test_hardware_detection.py` | Unit | Mock GPU detection, VRAM calculation |
| `test_model_loader.py` | Integration | Load each model variant, verify singleton |

#### Deliverables
- Working model loader that can load any Qwen3-VL variant
- Configuration system with environment variable support
- Hardware detection with recommendations

---

### Phase 1 - Core Extraction

**Goal**: Basic OCR and text extraction with Gradio UI

#### Tasks
- [ ] Create base task handler abstract class
- [ ] Implement OCR handler (full-page, region-based)
- [ ] Implement layout analysis handler
- [ ] Add bounding box visualization
- [ ] Build basic Gradio UI
  - [ ] Image upload
  - [ ] Task selection dropdown
  - [ ] Streaming response display
  - [ ] Result visualization panel
- [ ] Implement parsers (JSON from markdown, coordinates)

#### Tests
| Test | Type | Description |
|------|------|-------------|
| `test_parsers.py` | Unit | JSON extraction, coordinate parsing |
| `test_visualization.py` | Unit | Bounding box drawing, colors |
| `test_ocr_handler.py` | Integration | OCR accuracy on test images |
| `test_layout_handler.py` | Integration | Section detection accuracy |
| `test_gradio_basic.py` | Integration | UI renders, file upload works |

#### Deliverables
- Working OCR with bounding boxes
- Layout detection (headers, sections, paragraphs)
- Gradio UI with image upload and visualization
- Output formats: plain text, JSON with coordinates

---

### Phase 2 - Structured Extraction

**Goal**: Schema-based field extraction, tables, and NER

#### Tasks
- [ ] Implement table extraction handler
  - [ ] Detect table boundaries
  - [ ] Extract rows/columns to JSON
  - [ ] Export to CSV format
- [ ] Implement field extraction with schemas
  - [ ] User-defined JSON schema input
  - [ ] Extract matching fields from document
  - [ ] Confidence scores per field
- [ ] Implement NER handler
  - [ ] Person names
  - [ ] Organizations
  - [ ] Dates/times
  - [ ] Monetary amounts
  - [ ] Locations
- [ ] Add schema definition UI in Gradio
  - [ ] Schema builder interface
  - [ ] Preset templates (invoice, receipt, ID)
  - [ ] Custom field definitions
- [ ] Format validation (dates, amounts, emails)

#### Tests
| Test | Type | Description |
|------|------|-------------|
| `test_table_extraction.py` | Integration | Table detection, row/column accuracy |
| `test_field_extraction.py` | Integration | Schema matching, confidence scores |
| `test_ner_handler.py` | Integration | Entity recognition accuracy |
| `test_schema_validation.py` | Unit | Schema parsing, field type validation |
| `test_format_validators.py` | Unit | Date, amount, email validation |

#### Deliverables
- Table to JSON/CSV conversion
- Schema-based field extraction
- NER with entity categorization
- Format validation for common types

---

### Phase 3 - Document Intelligence

**Goal**: Domain-specific document understanding

#### Tasks
- [ ] Implement form understanding handler
  - [ ] Key-value pair detection
  - [ ] Checkbox/radio button state
  - [ ] Signature detection
- [ ] Implement invoice/receipt parser
  - [ ] Vendor information
  - [ ] Line items with quantities/prices
  - [ ] Tax, subtotal, total calculation
  - [ ] Payment information
- [ ] Implement contract analyzer
  - [ ] Party identification
  - [ ] Key dates (effective, expiration)
  - [ ] Clause extraction
  - [ ] Obligation identification
- [ ] Add document type templates in UI
- [ ] Multi-page document support
  - [ ] Context merging across pages
  - [ ] Document boundary detection
- [ ] Cross-field validation
  - [ ] Total = sum of items
  - [ ] Date consistency

#### Tests
| Test | Type | Description |
|------|------|-------------|
| `test_form_handler.py` | Integration | Form field extraction, checkbox detection |
| `test_invoice_parser.py` | Integration | Line item accuracy, total verification |
| `test_contract_analyzer.py` | Integration | Party/date extraction, clause detection |
| `test_multipage.py` | Integration | Context continuity, boundary detection |
| `test_cross_validation.py` | Unit | Field relationship validation |

#### Deliverables
- Form understanding with key-value pairs
- Invoice/receipt structured output
- Contract analysis with clause extraction
- Multi-page document support
- Cross-field validation

---

### Phase 4 - Developer Platform

**Goal**: API generation, batch processing, integrations

#### Tasks
- [ ] Implement API generation from schemas
  - [ ] Generate Pydantic models from extraction schemas
  - [ ] Auto-generate FastAPI endpoints
  - [ ] OpenAPI documentation
- [ ] Build batch processing system
  - [ ] Folder upload support
  - [ ] Queue management
  - [ ] Progress tracking
  - [ ] Parallel processing (multiple documents)
- [ ] Add webhook integration
  - [ ] Configurable endpoints
  - [ ] Event types (completed, failed, progress)
  - [ ] Retry logic
- [ ] Storage integrations
  - [ ] S3/GCS upload support
  - [ ] Database write (PostgreSQL, MongoDB)
- [ ] Export functionality
  - [ ] PDF report generation
  - [ ] Excel/CSV export
  - [ ] JSON download

#### Tests
| Test | Type | Description |
|------|------|-------------|
| `test_api_generation.py` | Unit | Pydantic model generation from schema |
| `test_batch_processing.py` | Integration | Queue, progress, parallel execution |
| `test_webhooks.py` | Integration | Event delivery, retry logic |
| `test_storage.py` | Integration | S3/GCS upload, database write |
| `test_exports.py` | Unit | PDF, Excel, CSV generation |

#### Deliverables
- Auto-generated APIs from extraction schemas
- Batch processing with queue management
- Webhook notifications
- Cloud storage and database integrations
- Multiple export formats

---

### Phase 5 - Enterprise Features

**Goal**: Multi-tenant, monitoring, compliance

#### Tasks
- [ ] Implement multi-tenant architecture
  - [ ] User/organization isolation
  - [ ] Workspace management
  - [ ] Resource quotas
- [ ] Build monitoring dashboard
  - [ ] Request volume and latency
  - [ ] Error rates by task type
  - [ ] GPU utilization
  - [ ] Cost tracking (tokens, GPU hours)
  - [ ] Prometheus metrics export
- [ ] Implement audit logging
  - [ ] Who processed what, when
  - [ ] Input/output logging (configurable)
  - [ ] Compliance reports
- [ ] Add authentication/authorization
  - [ ] API key management
  - [ ] Role-based access control
  - [ ] Rate limiting per user/tier
- [ ] Deployment enhancements
  - [ ] Kubernetes manifests
  - [ ] Horizontal pod autoscaling
  - [ ] Load balancing

#### Tests
| Test | Type | Description |
|------|------|-------------|
| `test_multitenancy.py` | Integration | Isolation, quota enforcement |
| `test_monitoring.py` | Integration | Metrics collection, dashboard data |
| `test_audit_logging.py` | Integration | Log completeness, compliance |
| `test_auth.py` | Integration | API keys, RBAC, rate limits |
| `test_k8s_deployment.py` | Integration | Scaling, load distribution |

#### Deliverables
- Multi-tenant support with workspaces
- Monitoring dashboard with cost tracking
- Comprehensive audit logging
- Authentication and rate limiting
- Kubernetes deployment with autoscaling

---

## Phase Summary

| Phase | Focus | Key Deliverable | Est. Complexity |
|-------|-------|-----------------|-----------------|
| **0** | Foundation | Model loader, config | Low |
| **1** | Core Extraction | OCR + Layout + Gradio UI | Medium |
| **2** | Structured Extraction | Tables, NER, Schemas | Medium |
| **3** | Document Intelligence | Forms, Invoices, Contracts | High |
| **4** | Developer Platform | API gen, Batch, Webhooks | High |
| **5** | Enterprise | Multi-tenant, Monitoring | High |

---

## Test Plan

### Unit Tests (No GPU Required)

| Component | Test File | Test Cases |
|-----------|-----------|------------|
| **config.py** | `test_config.py` | - Load default config<br>- Override with env vars<br>- Validate required fields<br>- Invalid config handling |
| **utils/parsers.py** | `test_parsers.py` | - Parse JSON from markdown<br>- Parse XML coordinates<br>- Clean HTML output<br>- Handle malformed input |
| **utils/visualization.py** | `test_visualization.py` | - Draw bounding boxes<br>- Draw points<br>- Color selection<br>- Font loading |
| **utils/image_processing.py** | `test_image_processing.py` | - Smart resize<br>- Format conversion<br>- Pixel range validation |
| **schemas/requests.py** | `test_schemas.py` | - Valid request validation<br>- Missing fields<br>- Invalid types<br>- File path validation |
| **tasks/base.py** | `test_task_handlers.py` | - Handler registration<br>- System prompt loading<br>- Task type enum |

### Integration Tests (GPU Required)

| Component | Test File | Test Cases |
|-----------|-----------|------------|
| **model_loader.py** | `test_model_loader.py` | - Load 3B model<br>- Load 7B model<br>- Quantization config<br>- Device allocation<br>- Singleton behavior |
| **inference_engine.py** | `test_inference_engine.py` | - Initialize engine<br>- Single image inference<br>- Video inference<br>- Streaming output<br>- Error handling |
| **tasks/*.py** | `test_tasks.py` | - OCR accuracy<br>- Document parsing output<br>- Bounding box format<br>- Agent function calls<br>- Video frame handling |
| **gradio_app.py** | `test_gradio_app.py` | - UI component rendering<br>- File upload handling<br>- Chat history<br>- Task switching<br>- Streaming display |

### Test Commands

```bash
# Run all unit tests (CI-safe)
pytest tests/unit -v -m unit

# Run specific unit test
pytest tests/unit/test_parsers.py -v

# Run integration tests (requires GPU)
pytest tests/integration -v -m integration

# Run with coverage
pytest tests/unit --cov=qwen_vl --cov-report=html

# Run single module test
python -m qwen_vl.utils.parsers

# Smoke test
python -m qwen_vl.core.inference_engine --smoke-test
```

### Test Fixtures (conftest.py)

```python
@pytest.fixture
def sample_image():
    """Provide test image path."""
    return "tests/fixtures/sample.jpg"

@pytest.fixture
def sample_video():
    """Provide test video path."""
    return "tests/fixtures/sample.mp4"

@pytest.fixture
def mock_model():
    """Mock model for unit tests."""
    return MagicMock()

@pytest.fixture(scope="session")
def loaded_model():
    """Real model for integration tests."""
    pytest.importorskip("torch")
    from qwen_vl.core.model_loader import get_model
    return get_model()
```

---

## Configuration

### Environment Variables

```bash
# Model settings
QWEN_MODEL_SIZE=4B              # 2B, 4B, or 8B
QWEN_MODEL_VARIANT=instruct     # instruct or thinking
QWEN_QUANTIZATION=4bit          # none, 4bit, 8bit
QWEN_DEVICE_MAP=auto            # auto, cuda:0, cpu

# Inference settings
QWEN_MAX_NEW_TOKENS=4096
QWEN_MIN_PIXELS=401408          # 512 * 28 * 28
QWEN_MAX_PIXELS=1605632         # 2048 * 28 * 28

# Server settings
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json                 # json or text
```

### config.py Example

```python
from dataclasses import dataclass
from typing import Literal
import os

@dataclass
class ModelConfig:
    size: Literal["2B", "4B", "8B"] = "4B"
    variant: Literal["instruct", "thinking"] = "instruct"
    quantization: Literal["none", "4bit", "8bit"] = "4bit"
    device_map: str = "auto"

@dataclass
class InferenceConfig:
    max_new_tokens: int = 4096
    min_pixels: int = 512 * 28 * 28
    max_pixels: int = 2048 * 28 * 28
    total_pixels: int = 20480 * 28 * 28

@dataclass
class Config:
    model: ModelConfig
    inference: InferenceConfig

def load_config() -> Config:
    return Config(
        model=ModelConfig(
            size=os.getenv("QWEN_MODEL_SIZE", "4B"),
            variant=os.getenv("QWEN_MODEL_VARIANT", "instruct"),
            quantization=os.getenv("QWEN_QUANTIZATION", "4bit"),
        ),
        inference=InferenceConfig(
            max_new_tokens=int(os.getenv("QWEN_MAX_NEW_TOKENS", 4096)),
        ),
    )
```

---

## Deployment

### Docker

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY qwen_vl/ ./qwen_vl/
COPY main.py config.py ./

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:7860/ || exit 1

# Run application
CMD ["python", "main.py"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  qwen-vl:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - QWEN_MODEL_SIZE=4B
      - QWEN_MODEL_VARIANT=instruct
      - QWEN_QUANTIZATION=4bit
      - LOG_LEVEL=INFO
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Quick Start

```bash
# Clone and setup
git clone <repository>
cd qwen_vl

# Install dependencies
pip install -r requirements.txt

# Run with default settings
python main.py

# Run with Docker
docker-compose up --build
```

---

## Contributing

1. Follow the workflow in [WORKFLOW.md](WORKFLOW.md)
2. Create feature branch: `claude/<feature>-<session-id>`
3. Write tests for new features
4. Ensure all unit tests pass: `pytest tests/unit -v`
5. Submit PR with description

---

## License

[Specify License]

---

## Acknowledgments

- Qwen3-VL model by Alibaba Cloud
- Hugging Face Transformers
- Gradio by Hugging Face
