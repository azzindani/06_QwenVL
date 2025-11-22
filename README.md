# Qwen2.5-VL Production Deployment

A production-ready Vision-Language model service built on Qwen2.5-VL, featuring multi-task inference capabilities with a Gradio UI interface.

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

### Core Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| **OCR** | Full-page and line-level text extraction with bounding boxes | Planned |
| **Document Parsing** | Extract structured HTML from documents | Planned |
| **Image Recognition** | General image understanding and description | Planned |
| **Spatial Understanding** | Object detection with 2D bounding boxes | Planned |
| **Video Understanding** | Temporal analysis and description | Planned |
| **Computer Agent** | Function calling for UI automation | Planned |
| **Mobile Agent** | Mobile interface control | Planned |

### Production Features

| Feature | Description | Priority |
|---------|-------------|----------|
| Model quantization (4-bit) | Reduce VRAM usage | MVP |
| Streaming responses | Real-time token generation | MVP |
| Multi-GPU support | Distribute model across GPUs | MVP |
| Gradio UI | Web interface for all tasks | MVP |
| Configuration management | Centralized YAML config | MVP |
| Structured logging | JSON logs for aggregation | Phase 1 |
| Health checks | Readiness/liveness probes | Phase 1 |
| Request validation | Pydantic models | Phase 1 |
| Docker deployment | Containerization | Phase 1 |
| Batch inference | Process multiple inputs | Phase 2 |
| Caching | Model and result caching | Phase 2 |
| Metrics/monitoring | Prometheus metrics | Phase 2 |

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
| **Inference latency** | < 5s | Single image, 512 tokens |
| **Throughput** | 10 req/min | Single GPU (RTX 3090/4090) |
| **Memory usage** | < 12GB VRAM | 3B model with 4-bit quantization |
| **Memory usage** | < 20GB VRAM | 7B model with 4-bit quantization |

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

### MVP - Minimum Viable Product

**Goal**: Single working endpoint with core inference

- [ ] **Project setup**
  - [ ] Initialize directory structure
  - [ ] Create config.py with all settings
  - [ ] Set up logging infrastructure
  - [ ] Create requirements.txt

- [ ] **Core infrastructure**
  - [ ] Implement hardware_detection.py
  - [ ] Implement model_loader.py with singleton pattern
  - [ ] Create base task handler abstract class

- [ ] **Inference engine**
  - [ ] Port UnifiedQwenInference from notebooks
  - [ ] Implement streaming support
  - [ ] Add input validation

- [ ] **Task handlers (minimum 3)**
  - [ ] OCR handler
  - [ ] Image recognition handler
  - [ ] Document parsing handler

- [ ] **Gradio UI (basic)**
  - [ ] Single page with chat interface
  - [ ] Image upload support
  - [ ] Task type selection
  - [ ] Streaming response display

- [ ] **Testing**
  - [ ] Unit tests for utils
  - [ ] Unit tests for schemas
  - [ ] Manual integration testing

### Phase 1 - Production Ready

**Goal**: Deployable with CI/CD and monitoring

- [ ] **Complete all task handlers**
  - [ ] Spatial understanding
  - [ ] Video understanding
  - [ ] Computer agent
  - [ ] Mobile agent

- [ ] **Enhanced Gradio UI**
  - [ ] Multi-tab interface
  - [ ] Video upload support
  - [ ] Visualization panel
  - [ ] Settings/configuration panel
  - [ ] Chat history management

- [ ] **Production features**
  - [ ] Health check endpoints
  - [ ] Structured JSON logging
  - [ ] Request/response logging
  - [ ] Error tracking

- [ ] **Deployment**
  - [ ] Dockerfile with GPU support
  - [ ] docker-compose.yml
  - [ ] Environment variable documentation
  - [ ] .env.example

- [ ] **CI/CD**
  - [ ] GitHub Actions workflow
  - [ ] Unit test automation
  - [ ] Linting (black, isort, flake8)
  - [ ] Docker build verification

- [ ] **Testing**
  - [ ] Integration tests for all tasks
  - [ ] Gradio UI tests
  - [ ] Load testing basics

- [ ] **Documentation**
  - [ ] API documentation
  - [ ] Deployment guide
  - [ ] Configuration reference

### Phase 2 - Enhanced Features

**Goal**: Optimized performance and user experience

- [ ] **Performance optimization**
  - [ ] Model caching strategies
  - [ ] Batch inference support
  - [ ] Response caching
  - [ ] Memory optimization

- [ ] **Advanced UI features**
  - [ ] Multiple conversation sessions
  - [ ] Export results (PDF, JSON)
  - [ ] Custom system prompts
  - [ ] Theme customization

- [ ] **Monitoring & observability**
  - [ ] Prometheus metrics
  - [ ] Request tracing
  - [ ] Performance dashboards
  - [ ] Alerting

- [ ] **Multi-model support**
  - [ ] Dynamic model switching (3B/7B)
  - [ ] Model comparison mode

### Phase 3 - Advanced

**Goal**: Enterprise-ready features

- [ ] **Scalability**
  - [ ] Kubernetes deployment
  - [ ] Horizontal scaling
  - [ ] Load balancing

- [ ] **Enterprise features**
  - [ ] Authentication/authorization
  - [ ] Rate limiting
  - [ ] Usage analytics
  - [ ] Audit logging

- [ ] **Advanced capabilities**
  - [ ] Fine-tuning interface
  - [ ] Custom task creation
  - [ ] Plugin system

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
QWEN_MODEL_SIZE=3B              # 3B or 7B
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
    size: Literal["3B", "7B"] = "3B"
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
            size=os.getenv("QWEN_MODEL_SIZE", "3B"),
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
      - QWEN_MODEL_SIZE=3B
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

- Qwen2.5-VL model by Alibaba Cloud
- Hugging Face Transformers
- Gradio by Hugging Face
