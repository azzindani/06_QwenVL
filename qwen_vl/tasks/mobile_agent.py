"""Mobile agent task handler for mobile app automation."""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from PIL import Image, ImageColor, ImageDraw

from .base import BaseTaskHandler, TaskResult, TaskType, register_handler

logger = logging.getLogger(__name__)


def draw_touch_point(
    image: Image.Image,
    point: List[int],
    color: str = "green",
) -> Image.Image:
    """
    Draw a touch point visualization on an image.
    
    Args:
        image: PIL Image to draw on
        point: [x, y] coordinates
        color: Color for the point
        
    Returns:
        Image with drawn point
    """
    try:
        rgba_color = ImageColor.getrgb(color)
        if len(rgba_color) == 3:
            rgba_color = rgba_color + (128,)
    except ValueError:
        rgba_color = (0, 255, 0, 128)

    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    radius = min(image.size) * 0.05
    x, y = point

    # Draw semi-transparent touch circle
    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=rgba_color,
    )

    image = image.convert("RGBA")
    combined = Image.alpha_composite(image, overlay)

    return combined.convert("RGB")


def parse_mobile_action(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse mobile action from model response.
    
    Args:
        response: Model response text
        
    Returns:
        Parsed action dictionary or None
    """
    try:
        if "<tool_call>" in response and "</tool_call>" in response:
            tool_content = response.split("<tool_call>\n")[1].split("\n</tool_call>")[0]
            return json.loads(tool_content)
    except (IndexError, json.JSONDecodeError):
        pass
    return None


@register_handler(TaskType.MOBILE_AGENT)
class MobileAgentHandler(BaseTaskHandler):
    """Handler for mobile use agent tasks (mobile app automation)."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.MOBILE_AGENT

    @property
    def system_prompt(self) -> str:
        return (
            "You are a helpful mobile assistant. Analyze mobile screenshots and help users "
            "interact with their mobile device by identifying UI elements and suggesting actions. "
            "When an action is needed, describe what should be tapped, swiped, or typed."
        )

    def process(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        **kwargs,
    ) -> TaskResult:
        """
        Analyze a mobile screenshot and suggest actions.

        Args:
            image: Screenshot path or PIL Image
            prompt: User's task or query

        Returns:
            TaskResult with suggested actions
        """
        img = self._load_image(image)

        user_prompt = prompt or "What can you see on this mobile screen? Describe the main elements."

        messages = self._build_messages(img, user_prompt)
        response = self._generate(messages, **kwargs)

        # Try to parse any action
        action = parse_mobile_action(response)

        # Visualize tap if present
        vis_image = None
        if action and "arguments" in action:
            args = action["arguments"]
            if "coordinate" in args and args.get("action") == "click":
                vis_image = draw_touch_point(img, args["coordinate"], color="green")

        return TaskResult(
            text=response,
            data=action,
            visualization=vis_image,
            metadata={"mode": "mobile_agent"},
        )

    def find_element(
        self,
        image: Union[str, Image.Image],
        element_description: str,
        **kwargs,
    ) -> TaskResult:
        """
        Find a UI element on mobile screen.

        Args:
            image: Screenshot path or PIL Image
            element_description: Description of element to find

        Returns:
            TaskResult with element location
        """
        img = self._load_image(image)

        prompt = (
            f"Find the {element_description} on this mobile screen. "
            f"Describe its location and what gesture would interact with it."
        )

        messages = self._build_messages(img, prompt)
        response = self._generate(messages, **kwargs)

        return TaskResult(
            text=response,
            metadata={"mode": "find_element", "target": element_description},
        )

    def suggest_action(
        self,
        image: Union[str, Image.Image],
        task: str,
        history: Optional[str] = None,
        **kwargs,
    ) -> TaskResult:
        """
        Suggest the next action to complete a task on mobile.

        Args:
            image: Current screenshot
            task: Task to complete
            history: Previous action history (optional)

        Returns:
            TaskResult with suggested action
        """
        img = self._load_image(image)

        if history:
            prompt = (
                f"The user query: {task}\n"
                f"Task progress (You have done the following operation on the current device): {history}"
            )
        else:
            prompt = f"The user query: {task}\n\nWhat is the first action to complete this task?"

        messages = self._build_messages(img, prompt)
        response = self._generate(messages, **kwargs)

        action = parse_mobile_action(response)

        # Visualize if tap action
        vis_image = None
        if action and "arguments" in action:
            args = action["arguments"]
            if "coordinate" in args and args.get("action") == "click":
                vis_image = draw_touch_point(img, args["coordinate"])

        return TaskResult(
            text=response,
            data=action,
            visualization=vis_image,
            metadata={"mode": "suggest_action", "task": task},
        )

    def describe_screen(
        self,
        image: Union[str, Image.Image],
        **kwargs,
    ) -> TaskResult:
        """
        Describe the current mobile screen state.

        Args:
            image: Screenshot path or PIL Image

        Returns:
            TaskResult with screen description
        """
        img = self._load_image(image)

        prompt = (
            "Describe this mobile screen in detail. Include:\n"
            "1. What app appears to be open\n"
            "2. Main UI elements visible\n"
            "3. Any buttons, text fields, or interactive elements\n"
            "4. The current state or context"
        )

        messages = self._build_messages(img, prompt)
        response = self._generate(messages, **kwargs)

        return TaskResult(
            text=response,
            metadata={"mode": "describe_screen"},
        )


if __name__ == "__main__":
    print("=" * 60)
    print("MOBILE AGENT HANDLER TEST")
    print("=" * 60)
    print("  Mobile agent handler registered successfully")
    print(f"  Task type: {TaskType.MOBILE_AGENT}")
    print("  Note: Full agent functionality requires 'qwen-agent' library")
    print("  (Actual inference requires loaded model)")
    print("=" * 60)
