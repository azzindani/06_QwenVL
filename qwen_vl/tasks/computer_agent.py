"""Computer agent task handler for desktop automation."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageColor, ImageDraw

from .base import BaseTaskHandler, TaskResult, TaskType, register_handler

logger = logging.getLogger(__name__)


def draw_click_point(
    image: Image.Image,
    point: List[int],
    color: str = "red",
) -> Image.Image:
    """
    Draw a click point visualization on an image.
    
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
        rgba_color = (255, 0, 0, 128)

    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    radius = min(image.size) * 0.05
    x, y = point

    # Draw semi-transparent circle
    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=rgba_color,
    )

    # Draw center point
    center_radius = radius * 0.1
    overlay_draw.ellipse(
        [(x - center_radius, y - center_radius), (x + center_radius, y + center_radius)],
        fill=(0, 255, 0, 255),
    )

    image = image.convert("RGBA")
    combined = Image.alpha_composite(image, overlay)

    return combined.convert("RGB")


def parse_tool_call(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse tool call from model response.
    
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


@register_handler(TaskType.COMPUTER_AGENT)
class ComputerAgentHandler(BaseTaskHandler):
    """Handler for computer use agent tasks (desktop automation)."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.COMPUTER_AGENT

    @property
    def system_prompt(self) -> str:
        return (
            "You are a helpful computer use assistant. Analyze screenshots and help users "
            "interact with their computer by identifying UI elements and suggesting actions. "
            "When an action is needed, describe what should be clicked or typed."
        )

    def process(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        **kwargs,
    ) -> TaskResult:
        """
        Analyze a screenshot and suggest actions.

        Args:
            image: Screenshot path or PIL Image
            prompt: User's task or query

        Returns:
            TaskResult with suggested actions
        """
        img = self._load_image(image)

        user_prompt = prompt or "What can you see on this screen? Describe the main elements."

        messages = self._build_messages(img, user_prompt)
        response = self._generate(messages, **kwargs)

        # Try to parse any tool calls
        action = parse_tool_call(response)

        # Visualize click if present
        vis_image = None
        if action and "arguments" in action:
            args = action["arguments"]
            if "coordinate" in args and "click" in str(args.get("action", "")):
                vis_image = draw_click_point(img, args["coordinate"], color="green")

        return TaskResult(
            text=response,
            data=action,
            visualization=vis_image,
            metadata={"mode": "computer_agent"},
        )

    def find_element(
        self,
        image: Union[str, Image.Image],
        element_description: str,
        **kwargs,
    ) -> TaskResult:
        """
        Find a UI element on screen.

        Args:
            image: Screenshot path or PIL Image
            element_description: Description of element to find

        Returns:
            TaskResult with element location
        """
        img = self._load_image(image)

        prompt = (
            f"Find the {element_description} on this screen. "
            f"Describe its location and what action would interact with it."
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
        Suggest the next action to complete a task.

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
                f"Task: {task}\n\n"
                f"Previous actions: {history}\n\n"
                f"What is the next action to complete this task?"
            )
        else:
            prompt = f"Task: {task}\n\nWhat is the first action to complete this task?"

        messages = self._build_messages(img, prompt)
        response = self._generate(messages, **kwargs)

        action = parse_tool_call(response)

        return TaskResult(
            text=response,
            data=action,
            metadata={"mode": "suggest_action", "task": task},
        )


if __name__ == "__main__":
    print("=" * 60)
    print("COMPUTER AGENT HANDLER TEST")
    print("=" * 60)
    print("  Computer agent handler registered successfully")
    print(f"  Task type: {TaskType.COMPUTER_AGENT}")
    print("  Note: Full agent functionality requires 'qwen-agent' library")
    print("  (Actual inference requires loaded model)")
    print("=" * 60)
