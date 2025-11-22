"""Base task handler abstract class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from PIL import Image


class TaskType(str, Enum):
    """Available task types for document processing."""

    OCR = "ocr"
    LAYOUT = "layout"
    TABLE = "table"
    FIELD_EXTRACTION = "field_extraction"
    NER = "ner"
    FORM = "form"
    INVOICE = "invoice"
    CONTRACT = "contract"


@dataclass
class TaskResult:
    """Result from a task handler."""

    text: str
    data: Optional[Dict[str, Any]] = None
    bounding_boxes: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    visualization: Optional[Image.Image] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseTaskHandler(ABC):
    """Abstract base class for all task handlers."""

    def __init__(self, model: Any, processor: Any):
        """
        Initialize task handler.

        Args:
            model: Loaded Qwen3-VL model
            processor: Model processor/tokenizer
        """
        self.model = model
        self.processor = processor

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Return the task type this handler processes."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this task."""
        pass

    @abstractmethod
    def process(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        **kwargs,
    ) -> TaskResult:
        """
        Process an image with this task.

        Args:
            image: Image path or PIL Image
            prompt: Optional user prompt
            **kwargs: Additional task-specific arguments

        Returns:
            TaskResult with extracted information
        """
        pass

    def _load_image(self, image: Union[str, Image.Image]) -> Image.Image:
        """Load image from path or return if already PIL Image."""
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def _build_messages(
        self,
        image: Image.Image,
        user_prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        Build chat messages for the model.

        Args:
            image: PIL Image
            user_prompt: User's prompt/question

        Returns:
            List of message dicts for chat template
        """
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return messages

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate response from the model.

        Args:
            messages: Chat messages
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        import torch

        # Try to use official qwen_vl_utils for proper image/video processing
        try:
            from qwen_vl_utils import process_vision_info
            use_vision_utils = True
        except ImportError:
            use_vision_utils = False

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs using official pattern
        if use_vision_utils:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
            )
        else:
            # Fallback for when qwen_vl_utils is not available
            inputs = self.processor(
                text=[text],
                images=[msg["content"][0]["image"] for msg in messages if msg["role"] == "user"],
                return_tensors="pt",
                padding=True,
            )

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                **kwargs,
            )

        # Decode
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)

        return response


# Task handler registry
_handlers: Dict[TaskType, type] = {}


def register_handler(task_type: TaskType):
    """Decorator to register a task handler."""
    def decorator(cls):
        _handlers[task_type] = cls
        return cls
    return decorator


def get_handler(task_type: TaskType, model: Any, processor: Any) -> BaseTaskHandler:
    """
    Get a task handler instance.

    Args:
        task_type: Type of task
        model: Loaded model
        processor: Model processor

    Returns:
        Task handler instance
    """
    if task_type not in _handlers:
        raise ValueError(f"No handler registered for task type: {task_type}")

    return _handlers[task_type](model, processor)


def list_handlers() -> List[TaskType]:
    """List all registered task handlers."""
    return list(_handlers.keys())
