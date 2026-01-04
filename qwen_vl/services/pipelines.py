"""
Processing pipelines for chaining multiple tasks.

Provides:
- Pipeline class for chaining tasks
- Async execution support
- Progress tracking
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from PIL import Image

from ..tasks import TaskType, get_handler
from ..tasks.base import TaskResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineStage:
    """A single stage in the pipeline."""
    name: str
    task_type: TaskType
    options: Dict[str, Any] = field(default_factory=dict)
    transform: Optional[Callable[[TaskResult], Any]] = None


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    status: PipelineStatus
    stages: List[Dict[str, Any]]
    final_result: Optional[TaskResult] = None
    error: Optional[str] = None
    total_time: float = 0.0


class Pipeline:
    """
    Processing pipeline for chaining multiple tasks.
    
    Example usage:
        pipeline = Pipeline(model, processor)
        pipeline.add_stage("ocr", TaskType.OCR)
        pipeline.add_stage("ner", TaskType.NER)
        result = pipeline.run(image)
    """
    
    def __init__(self, model, processor):
        """
        Initialize pipeline.
        
        Args:
            model: Loaded model instance
            processor: Loaded processor instance
        """
        self.model = model
        self.processor = processor
        self.stages: List[PipelineStage] = []
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None
    
    def add_stage(
        self,
        name: str,
        task_type: TaskType,
        options: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable[[TaskResult], Any]] = None,
    ) -> "Pipeline":
        """
        Add a stage to the pipeline.
        
        Args:
            name: Stage name for identification
            task_type: Type of task to run
            options: Options to pass to the handler
            transform: Optional function to transform the result
            
        Returns:
            Self for chaining
        """
        stage = PipelineStage(
            name=name,
            task_type=task_type,
            options=options or {},
            transform=transform,
        )
        self.stages.append(stage)
        logger.debug(f"Added pipeline stage: {name} ({task_type.value})")
        return self
    
    def on_progress(
        self,
        callback: Callable[[int, int, str], None],
    ) -> "Pipeline":
        """
        Set progress callback.
        
        Args:
            callback: Function(current, total, stage_name)
            
        Returns:
            Self for chaining
        """
        self._progress_callback = callback
        return self
    
    def run(
        self,
        image: Union[str, Image.Image],
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Run the pipeline.
        
        Args:
            image: Image to process
            context: Optional context passed to all stages
            
        Returns:
            PipelineResult with stage results
        """
        import time
        
        start_time = time.time()
        stage_results = []
        current_result = None
        ctx = context or {}
        
        logger.info(f"Starting pipeline with {len(self.stages)} stages")
        
        for i, stage in enumerate(self.stages):
            stage_start = time.time()
            
            try:
                # Report progress
                if self._progress_callback:
                    self._progress_callback(i + 1, len(self.stages), stage.name)
                
                logger.info(f"Running stage {i+1}/{len(self.stages)}: {stage.name}")
                
                # Get handler
                handler = get_handler(
                    stage.task_type,
                    self.model,
                    self.processor,
                )
                
                # Build options from stage config and context
                options = {**stage.options}
                if current_result and "previous_result" not in options:
                    options["previous_result"] = current_result
                
                # Run the stage
                result = handler.process(image, **options)
                
                # Apply transform if provided
                if stage.transform:
                    transformed = stage.transform(result)
                    ctx[stage.name] = transformed
                else:
                    ctx[stage.name] = result
                
                current_result = result
                
                stage_time = time.time() - stage_start
                stage_results.append({
                    "name": stage.name,
                    "status": "completed",
                    "time": stage_time,
                    "metadata": result.metadata,
                })
                
                logger.info(f"Stage {stage.name} completed in {stage_time:.2f}s")
                
            except Exception as e:
                stage_time = time.time() - stage_start
                stage_results.append({
                    "name": stage.name,
                    "status": "failed",
                    "time": stage_time,
                    "error": str(e),
                })
                
                logger.error(f"Stage {stage.name} failed: {e}")
                
                return PipelineResult(
                    status=PipelineStatus.FAILED,
                    stages=stage_results,
                    error=str(e),
                    total_time=time.time() - start_time,
                )
        
        total_time = time.time() - start_time
        logger.info(f"Pipeline completed in {total_time:.2f}s")
        
        return PipelineResult(
            status=PipelineStatus.COMPLETED,
            stages=stage_results,
            final_result=current_result,
            total_time=total_time,
        )


# Pre-built pipelines
def create_document_pipeline(model, processor) -> Pipeline:
    """
    Create a pipeline for full document processing.
    
    Stages: OCR -> NER -> Summary
    """
    return (
        Pipeline(model, processor)
        .add_stage("ocr", TaskType.OCR, {"with_boxes": True})
        .add_stage("ner", TaskType.NER)
        .add_stage("layout", TaskType.LAYOUT)
    )


def create_invoice_pipeline(model, processor) -> Pipeline:
    """
    Create a pipeline for invoice processing.
    
    Stages: OCR -> Table -> Invoice
    """
    return (
        Pipeline(model, processor)
        .add_stage("ocr", TaskType.OCR)
        .add_stage("table", TaskType.TABLE)
        .add_stage("invoice", TaskType.INVOICE)
    )


if __name__ == "__main__":
    print("="*60)
    print("PIPELINE MODULE TEST")
    print("="*60)
    print("  Pipeline classes available:")
    print("  - Pipeline")
    print("  - PipelineStage")
    print("  - PipelineResult")
    print("  Pre-built pipelines:")
    print("  - create_document_pipeline()")
    print("  - create_invoice_pipeline()")
    print("="*60)
