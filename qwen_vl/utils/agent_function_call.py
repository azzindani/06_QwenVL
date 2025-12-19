"""Agent tool definitions for computer and mobile interaction."""

from typing import Any, Dict, Optional


class ComputerUse:
    """Tool definition for computer interaction."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        Initialize the tool.
        
        Args:
            cfg: Configuration including display_width_px and display_height_px.
        """
        self.cfg = cfg or {}
        width = self.cfg.get("display_width_px", 1000)
        height = self.cfg.get("display_height_px", 1000)

        self.function = {
            "name": "computer_use",
            "description": (
                f"Interact with a computer (screen resolution: {width}x{height}). "
                "Can perform actions like clicking, typing, and opening applications."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["click", "type", "scroll", "wait", "open"],
                        "description": "The action to perform on the computer."
                    },
                    "coordinate": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "[x, y] coordinates for click/tap. Scale is 0-1000."
                    },
                    "text": {
                        "type": "string",
                        "description": "The text to type or the application to open."
                    },
                    "element": {
                        "type": "string",
                        "description": "A description of the element to interact with."
                    }
                },
                "required": ["action"]
            }
        }


class MobileUse:
    """Tool definition for mobile interaction."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        Initialize the tool.
        
        Args:
            cfg: Configuration including display_width_px and display_height_px.
        """
        self.cfg = cfg or {}
        width = self.cfg.get("display_width_px", 1000)
        height = self.cfg.get("display_height_px", 1000)

        self.function = {
            "name": "mobile_use",
            "description": (
                f"Interact with a mobile device (screen resolution: {width}x{height}). "
                "Can perform actions like clicking, swiping, and typing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["click", "swipe", "type", "wait", "open", "back"],
                        "description": "The action to perform on the mobile device."
                    },
                    "coordinate": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "[x, y] coordinates for click/tap. Scale is 0-1000."
                    },
                    "text": {
                        "type": "string",
                        "description": "The text to type or the application to open."
                    },
                    "element": {
                        "type": "string",
                        "description": "A description of the element to interact with."
                    }
                },
                "required": ["action"]
            }
        }
