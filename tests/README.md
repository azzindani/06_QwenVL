# Qwen VL Tests

This directory contains all tests for the Qwen VL project.

## Directory Structure

```
tests/
├── asset/                    # Sample test files (55 files)
│   ├── ocr_example*.jpg      # OCR test images (6)
│   ├── docparsing_example*   # Document parsing samples (8)
│   ├── computer_use*.jpeg    # Desktop screenshots (2)
│   ├── mobile_*_example.png  # Mobile screenshots (2)
│   ├── spatio_case*.png/jpg  # Spatial understanding (5)
│   ├── sample-celebrity*.jpg # Celebrity recognition (3)
│   ├── sample-bird.jpg       # Animal recognition
│   ├── sample-food.jpeg      # Food recognition
│   ├── lots_of_cars.png      # Vehicle detection
│   ├── lots_of_people.jpeg   # People detection
│   └── ...                   # More samples
│
├── unit/                     # Unit tests (no GPU required)
│
├── integration/              # Integration tests (GPU required)
│   ├── test_ocr.py           # OCR handler tests
│   ├── test_layout.py        # Layout handler tests
│   ├── test_recognition.py   # Recognition handler tests
│   ├── test_spatial.py       # Spatial handler tests
│   ├── test_document_parsing.py  # Document parsing tests
│   ├── test_computer_agent.py    # Computer agent tests
│   ├── test_mobile_agent.py      # Mobile agent tests
│   ├── test_extraction.py    # Table/NER/Form/Invoice/Contract
│   └── test_all_handlers.py  # Comprehensive test
│
└── results/                  # Test output (auto-generated)
```

---

## Quick Start

```bash
# 1. Activate environment
conda activate omni_env

# 2. Navigate to project
cd "d:\AI_Workspace\90_AI_Implementation\New folder\06_QwenVL"

# 3. Run a test
python tests/integration/test_ocr.py
```

python tests/integration/test_ocr.py
```

### 4. Run in Jupyter Notebook
To see image previews, use `%run` instead of `!python`:

```python
%run tests/integration/test_ocr.py
```

---

## Running Tests in Notebook (Jupyter/Kaggle/Colab)

To see inline "Before" and "After" image previews, you must run the tests inside the notebook kernel.
Running as a subprocess (e.g., `!python tests/...`) will **NOT** show images because the subprocess cannot modify the notebook display.

### Correct Way (Magic Command) - RECOMMENDED
The simplest way to run tests in a notebook cell with image previews is using the `%run` magic command. This runs the script in the current kernel context.

```python
%run tests/integration/test_ocr.py
```

### Alternative Way (Import)
Import the `main` function and run it:

```python
# 1. OCR Test
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from tests.integration.test_ocr import main
main()

# 2. Recognition Test
from tests.integration.test_recognition import main
main()

# 3. Spatial Test
from tests.integration.test_spatial import main
main()

# 4. Computer Agent Test
from tests.integration.test_computer_agent import main
main()

# 5. Mobile Agent Test
from tests.integration.test_mobile_agent import main
main()

# 6. Document Parsing Test
from tests.integration.test_document_parsing import main
main()

# 7. Layout Analysis Test
from tests.integration.test_layout import main
main()

# 8. Extraction Test
from tests.integration.test_extraction import main
main()
```

---

## Running Individual Feature Tests

Each feature has its own test file with dedicated samples:

| Test File | Samples Used | What It Tests |
|-----------|--------------|---------------|
| `test_ocr.py` | `ocr_example1-6.jpg` | Basic OCR, with boxes, line extraction |
| `test_layout.py` | `docparsing_example*.jpg/png` | Layout analysis, sections, reading order |
| `test_recognition.py` | `sample-celebrity*.jpg`, `sample-bird.jpg`, `sample-food.jpeg`, scenes | Celebrity, animal, food, scene recognition |
| `test_spatial.py` | `spatio_case*.png`, `lots_of_cars.png`, `lots_of_people.jpeg` | Object detection, car/people counting |
| `test_document_parsing.py` | `docparsing_example*.jpg/png` | HTML parsing, layout boxes |
| `test_computer_agent.py` | `computer_use1.jpeg`, `computer_use2.jpeg` | Screen analysis, find elements, suggest actions |
| `test_mobile_agent.py` | `mobile_en_example.png`, `mobile_zh_example.png` | Mobile screen analysis (EN & ZH) |
| `test_extraction.py` | `docparsing_example*.jpg` | Table, NER, Form, Invoice, Contract |

### Run Examples:

```bash
# OCR testing
python tests/integration/test_ocr.py
# Jupyter: %run tests/integration/test_ocr.py


# Recognition (celebrities, animals, food, scenes)
python tests/integration/test_recognition.py
# Jupyter: %run tests/integration/test_recognition.py

# Spatial understanding (detect cars, people, objects)
python tests/integration/test_spatial.py
# Jupyter: %run tests/integration/test_spatial.py

# Computer agent (desktop automation)
python tests/integration/test_computer_agent.py
# Jupyter: %run tests/integration/test_computer_agent.py

# Mobile agent (English and Chinese screens)
python tests/integration/test_mobile_agent.py
# Jupyter: %run tests/integration/test_mobile_agent.py

# Document parsing (HTML extraction)
python tests/integration/test_document_parsing.py
# Jupyter: %run tests/integration/test_document_parsing.py

# Layout analysis
python tests/integration/test_layout.py
# Jupyter: %run tests/integration/test_layout.py

# All extraction handlers (Table, NER, Form, Invoice, Contract)
python tests/integration/test_extraction.py
# Jupyter: %run tests/integration/test_extraction.py
```

---

## Sample Files Reference

### OCR Samples
- `ocr_example1.jpg` - `ocr_example6.jpg`: Various text documents

### Recognition Samples
- `sample-celebrity.jpeg`, `sample-celebrity-2.jpg`: Celebrity photos
- `sample-bird.jpg`: Bird/animal photo
- `sample-food.jpeg`: Food photo
- `sample-anime.jpeg`: Anime character
- `football_field.jpg`, `office.jpg`, `lounge.jpg`, `dining_table.png`: Scene photos

### Spatial Samples
- `spatio_case1.jpg`, `spatio_case2_*.png`: Spatial understanding cases
- `lots_of_cars.png`, `drone_cars2.png`: Vehicle detection
- `lots_of_people.jpeg`: People detection
- `autonomous_driving.jpg`: Driving scene

### Agent Samples
- `computer_use1.jpeg`, `computer_use2.jpeg`: Desktop screenshots
- `mobile_en_example.png`: English mobile app screen
- `mobile_zh_example.png`: Chinese mobile app screen
- `screenshot_demo.png`: Demo screenshot

### Document Samples
- `docparsing_example1.jpg` - `docparsing_example8.png`: Various document types

---

## Test Output

Results are saved to `tests/results/`:
- Visualization images with bounding boxes
- JSON test reports

### Notebook Visualization
All tests support inline "Before" (input) and "After" (result) visualization when run in Jupyter/Kaggle/Colab environments. This uses a robust display utility that works for both script and module execution.

---

## Troubleshooting

### Model Not Loading
```
❌ Failed to load model
```
- Check GPU: `nvidia-smi`
- Check VRAM: 3B needs ~6GB, 7B needs ~14GB

### No Samples Found
```
❌ No test images found
```
- Add images to `tests/asset/` directory
- Check file names match expected patterns

### Import Errors
```
ModuleNotFoundError: No module named 'qwen_vl'
```
- Run from project root directory
- Activate environment: `conda activate omni_env`
