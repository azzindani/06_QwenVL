"""Named Entity Recognition (NER) task handler."""

from typing import Any, Dict, List, Optional, Union

from PIL import Image

from ..utils.parsers import parse_json_from_markdown, parse_json_array_from_markdown, parse_entities_with_bbox, extract_malformed_bbox, normalize_bbox
from ..utils.visualization import draw_bounding_boxes
from .base import BaseTaskHandler, TaskResult, TaskType, register_handler


# Entity types with descriptions
ENTITY_TYPES = {
    "PERSON": "Names of people",
    "ORGANIZATION": "Companies, institutions, organizations",
    "LOCATION": "Places, addresses, geographic locations",
    "DATE": "Dates and times",
    "MONEY": "Monetary values and currencies",
    "EMAIL": "Email addresses",
    "PHONE": "Phone numbers",
    "URL": "Website URLs",
    "PRODUCT": "Product names",
    "EVENT": "Events",
}


@register_handler(TaskType.NER)
class NERHandler(BaseTaskHandler):
    """Handler for Named Entity Recognition tasks."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.NER

    @property
    def system_prompt(self) -> str:
        return (
            "You are a helpful assistant specialized in Named Entity Recognition (NER). "
            "Identify and extract named entities from text in images, categorizing them "
            "by type such as person names, organizations, dates, locations, and monetary values."
        )

    def process(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        **kwargs,
    ) -> TaskResult:
        """
        Extract named entities from an image.

        Args:
            image: Image path or PIL Image
            prompt: Optional custom prompt
            entity_types: List of entity types to extract (default: all)

        Returns:
            TaskResult with extracted entities
        """
        img = self._load_image(image)
        
        # Debug: Print original image size
        print(f"[NER] Original image size: {img.size}")

        # Filter entity types
        if entity_types:
            types_to_extract = {k: v for k, v in ENTITY_TYPES.items() if k in entity_types}
        else:
            types_to_extract = ENTITY_TYPES

        user_prompt = prompt or self._build_ner_prompt(types_to_extract)

        messages = self._build_messages(img, user_prompt)
        response = self._generate(messages, **kwargs)
        
        # Debug: Print model input dimensions
        print(f"[NER] Model input dimensions: ({self.last_input_width}x{self.last_input_height})")

        # Parse entities using robust parser that handles malformed JSON
        entities = parse_entities_with_bbox(response)
        
        # Fallback to standard parsing if robust parser returns empty
        if not entities:
            data = parse_json_from_markdown(response)
            entities = data.get("entities", []) if data else []
        
        print(f"[NER] Found {len(entities)} entities")

        # Group by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.get("type", "UNKNOWN")
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)

        # Create visualization
        boxes = []
        for entity in entities:
            # Check for bbox or bbox_2d
            bbox_raw = entity.get("bbox") or entity.get("bbox_2d")
            if bbox_raw:
                bbox = normalize_bbox(bbox_raw)
                if bbox:
                    boxes.append({
                        "bbox": bbox,
                        "label": f"{entity.get('type', '')}: {entity.get('text', '')[:15]}",
                    })
                    print(f"[NER] Entity bbox: {bbox} for '{entity.get('text', '')[:20]}'")
        
        print(f"[NER] Drawing {len(boxes)} boxes on image size {img.size}")

        vis_image = draw_bounding_boxes(
            img, 
            boxes,
            input_width=self.last_input_width,
            input_height=self.last_input_height
        ) if boxes else None

        return TaskResult(
            text=response,
            data={
                "entities": entities,
                "entities_by_type": entities_by_type,
            },
            bounding_boxes=boxes,
            visualization=vis_image,
            metadata={
                "entity_count": len(entities),
                "type_count": len(entities_by_type),
                "image_size": img.size,
                "model_input_size": (self.last_input_width, self.last_input_height),
            },
        )

    def _build_ner_prompt(self, entity_types: Dict[str, str]) -> str:
        """Build NER prompt from entity types."""
        type_descriptions = "\n".join([f"- {k}: {v}" for k, v in entity_types.items()])

        return (
            "Extract all named entities from this image. Identify:\n\n"
            f"{type_descriptions}\n\n"
            "Return as JSON array of entities:\n"
            "```json\n"
            '{\n'
            '  "entities": [\n'
            '    {\n'
            '      "text": "entity text",\n'
            '      "type": "ENTITY_TYPE",\n'
            '      "confidence": 0.95,\n'
            '      "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 50}\n'
            '    }\n'
            '  ]\n'
            '}\n'
            "```"
        )

    def extract_type(
        self,
        image: Union[str, Image.Image],
        entity_type: str,
        **kwargs,
    ) -> TaskResult:
        """
        Extract only a specific entity type.

        Args:
            image: Image path or PIL Image
            entity_type: Entity type to extract (e.g., 'PERSON', 'DATE')

        Returns:
            TaskResult with entities of specified type
        """
        return self.process(image, entity_types=[entity_type], **kwargs)

    @staticmethod
    def list_entity_types() -> Dict[str, str]:
        """List available entity types with descriptions."""
        return ENTITY_TYPES.copy()


if __name__ == "__main__":
    print("=" * 60)
    print("NER HANDLER TEST")
    print("=" * 60)
    print("  NER handler registered successfully")
    print(f"  Task type: {TaskType.NER}")
    print(f"  Entity types: {list(ENTITY_TYPES.keys())}")
    print("=" * 60)
