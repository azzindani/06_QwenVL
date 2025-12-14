"""Recognition task handler for image understanding."""

import logging
from typing import Optional, Union

from PIL import Image

from .base import BaseTaskHandler, TaskResult, TaskType, register_handler

logger = logging.getLogger(__name__)


@register_handler(TaskType.RECOGNITION)
class RecognitionHandler(BaseTaskHandler):
    """Handler for image recognition tasks (objects, landmarks, celebrities, etc.)."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.RECOGNITION

    @property
    def system_prompt(self) -> str:
        return (
            "You are a helpful assistant specialized in image recognition and understanding. "
            "Analyze images to identify objects, people, landmarks, animals, and other visual elements. "
            "Provide detailed and accurate descriptions."
        )

    def process(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        **kwargs,
    ) -> TaskResult:
        """
        Recognize and describe content in an image.

        Args:
            image: Image path or PIL Image
            prompt: Optional custom prompt

        Returns:
            TaskResult with recognition results
        """
        img = self._load_image(image)

        user_prompt = prompt or "What is the main subject of this image? Describe it in detail."

        messages = self._build_messages(img, user_prompt)
        response = self._generate(messages, **kwargs)

        return TaskResult(
            text=response,
            metadata={"mode": "recognition"},
        )

    def identify_objects(
        self,
        image: Union[str, Image.Image],
        **kwargs,
    ) -> TaskResult:
        """
        Identify all objects in the image.

        Args:
            image: Image path or PIL Image

        Returns:
            TaskResult with identified objects
        """
        img = self._load_image(image)

        prompt = (
            "List all objects visible in this image. "
            "For each object, provide its name and a brief description."
        )

        messages = self._build_messages(img, prompt)
        response = self._generate(messages, **kwargs)

        return TaskResult(
            text=response,
            metadata={"mode": "object_identification"},
        )

    def identify_landmarks(
        self,
        image: Union[str, Image.Image],
        **kwargs,
    ) -> TaskResult:
        """
        Identify landmarks in the image.

        Args:
            image: Image path or PIL Image

        Returns:
            TaskResult with landmark information
        """
        img = self._load_image(image)

        prompt = (
            "Identify any landmarks, famous buildings, or notable locations in this image. "
            "Provide their names in both English and Chinese if possible."
        )

        messages = self._build_messages(img, prompt)
        response = self._generate(messages, **kwargs)

        return TaskResult(
            text=response,
            metadata={"mode": "landmark_identification"},
        )


if __name__ == "__main__":
    print("=" * 60)
    print("RECOGNITION HANDLER TEST")
    print("=" * 60)
    print("  Recognition handler registered successfully")
    print(f"  Task type: {TaskType.RECOGNITION}")
    print("  (Actual inference requires loaded model)")
    print("=" * 60)
