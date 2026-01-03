"""Computer agent task handler for desktop automation."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageColor, ImageDraw

from .base import BaseTaskHandler, TaskResult, TaskType, register_handler
from ..utils.agent_function_call import ComputerUse

try:
    from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
        NousFnCallPrompt,
        Message,
        ContentItem,
    )
    HAS_QWEN_AGENT = True
except ImportError:
    HAS_QWEN_AGENT = False

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
            tool_content = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()
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
        return "You are a helpful assistant that can use a computer to solve tasks."

    def _build_agent_messages(
        self,
        image: Image.Image,
        user_prompt: str,
    ) -> List[Dict[str, Any]]:
        """Build messages using qwen_agent if available."""
        if not HAS_QWEN_AGENT:
            # Fallback to manual prompt construction if qwen_agent is missing
            system_p = (
                f"{self.system_prompt}\n\n"
                "## Tools\n"
                "You have access to the following tool:\n"
                "### computer_use\n"
                "Parameters:\n"
                "- action: (required, string) One of: 'click', 'type', 'scroll', 'wait', 'open'.\n"
                "- coordinate: (optional, array[integer]) [x, y] coordinates for click. Scale is 0-1000.\n"
                "- text: (optional, string) The text to type.\n"
                "- element: (optional, string) A description of the element.\n\n"
                "When you want to perform an action, you MUST use the following format:\n"
                "<tool_call>\n"
                "{\"name\": \"computer_use\", \"arguments\": {\"action\": \"...\", \"coordinate\": [...], \"element\": \"...\"}}\n"
                "</tool_call>"
            )
            return [
                {"role": "system", "content": system_p},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]

        # Use qwen_agent for robust prompt construction
        computer_use = ComputerUse(
            cfg={"display_width_px": image.width, "display_height_px": image.height}
        )
        prompt_builder = NousFnCallPrompt()
        
        # Set default pixel limits to avoid OOM for large screenshots
        min_pixels = 512 * 28 * 28
        max_pixels = 1024 * 28 * 28
        
        # Build standard Message objects
        messages = [
            Message(role="system", content=[ContentItem(text=self.system_prompt)]),
            Message(
                role="user",
                content=[
                    ContentItem(text=user_prompt),
                    ContentItem(image="image_placeholder"), # Use placeholder to avoid validation error
                ],
            ),
        ]
        
        # Preprocess with tool schemas
        processed_messages = prompt_builder.preprocess_fncall_messages(
            messages=messages,
            functions=[computer_use.function],
            lang=None,
        )
        
        # Convert back to dicts for our internal generator and fix image format
        final_messages = []
        for msg in processed_messages:
            msg_dict = msg.model_dump()
            fixed_content = []
            for item in msg_dict.get("content", []):
                if item.get("image") == "image_placeholder":
                    fixed_content.append({"type": "image", "image": image})
                elif item.get("text"):
                    fixed_content.append({"type": "text", "text": item["text"]})
                else:
                    fixed_content.append(item)
            msg_dict["content"] = fixed_content
            final_messages.append(msg_dict)
            
        return final_messages

    def process(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        **kwargs,
    ) -> TaskResult:
        """
        Analyze a screenshot and suggest actions with tool calls.

        Args:
            image: Screenshot path or PIL Image
            prompt: User's task or query

        Returns:
            TaskResult with suggested actions
        """
        img = self._load_image(image)

        user_prompt = prompt or "What can you see on this screen? If a task is implied, suggest the next step using a tool call."

        messages = self._build_agent_messages(img, user_prompt)
        response = self._generate(messages, **kwargs)

        # Try to parse any tool calls
        action = parse_tool_call(response)

        # Visualize action if present
        vis_image = None
        if action and "arguments" in action:
            args = action["arguments"]
            # Check for coordinates in either 'coordinate' or 'point'
            coord = args.get("coordinate") or args.get("point")
            
            if coord and isinstance(coord, list) and len(coord) == 2:
                # Some models might return click even if not explicitly in action name
                vis_image = draw_click_point(img, coord, color="green")
            elif "click" in str(args.get("action", "")).lower() and "coordinate" in args:
                vis_image = draw_click_point(img, args["coordinate"], color="green")

        return TaskResult(
            text=response,
            data=action,
            visualization=vis_image,
            metadata={"mode": "computer_agent", "has_tool_call": action is not None},
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
            f"Describe its location and provide a tool call to 'click' it if found."
        )

        messages = self._build_agent_messages(img, prompt)
        response = self._generate(messages, **kwargs)

        action = parse_tool_call(response)
        vis_image = None
        if action and "arguments" in action:
            args = action["arguments"]
            coord = args.get("coordinate") or args.get("point")
            if coord and isinstance(coord, list) and len(coord) == 2:
                vis_image = draw_click_point(img, coord, color="green")

        return TaskResult(
            text=response,
            data=action,
            visualization=vis_image,
            metadata={"mode": "find_element", "target": element_description, "has_tool_call": action is not None},
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
                f"What is the next action to complete this task? Suggest it using a tool call."
            )
        else:
            prompt = f"Task: {task}\n\nWhat is the first action to complete this task? Suggest it using a tool call."

        messages = self._build_agent_messages(img, prompt)
        response = self._generate(messages, **kwargs)

        action = parse_tool_call(response)
        vis_image = None
        if action and "arguments" in action:
            args = action["arguments"]
            coord = args.get("coordinate") or args.get("point")
            if coord and isinstance(coord, list) and len(coord) == 2:
                vis_image = draw_click_point(img, coord, color="green")

        return TaskResult(
            text=response,
            data=action,
            visualization=vis_image,
            metadata={"mode": "suggest_action", "task": task, "has_tool_call": action is not None},
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
