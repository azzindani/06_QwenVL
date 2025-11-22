"""Gradio UI for Qwen3-VL document processing."""

import logging
from typing import Generator, List, Optional, Tuple

import gradio as gr
from PIL import Image

from ..config import get_config
from ..core.model_loader import ModelLoader
from ..tasks import TaskType, get_handler, list_handlers

logger = logging.getLogger(__name__)


class GradioApp:
    """Gradio application for Qwen3-VL inference."""

    def __init__(self):
        self.model_loader = ModelLoader()
        self.config = get_config()
        self._model_loaded = False

    def load_model(self) -> str:
        """Load the model if not already loaded."""
        if self._model_loaded:
            return "Model already loaded"

        try:
            self.model_loader.load(self.config)
            self._model_loaded = True
            return f"Model loaded: {self.config.model.model_id}"
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return f"Failed to load model: {e}"

    def process_image(
        self,
        image: Optional[Image.Image],
        task_type: str,
        prompt: str,
        with_boxes: bool,
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, Optional[Image.Image]]:
        """
        Process an image with the selected task.

        Returns:
            Tuple of (text_result, visualization_image)
        """
        if image is None:
            return "Please upload an image", None

        if not self._model_loaded:
            return "Please load the model first", None

        try:
            # Get handler
            task = TaskType(task_type)
            loaded = self.model_loader._loaded_model
            handler = get_handler(task, loaded.model, loaded.processor)

            # Process
            kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            }

            if task == TaskType.OCR:
                kwargs["with_boxes"] = with_boxes

            if prompt.strip():
                result = handler.process(image, prompt=prompt, **kwargs)
            else:
                result = handler.process(image, **kwargs)

            return result.text, result.visualization

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return f"Error: {e}", None

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(
            title="Qwen2.5-VL Document Processing",
        ) as demo:
            gr.Markdown("# Qwen2.5-VL Document Processing")
            gr.Markdown("Upload an image and select a task to extract information.")

            with gr.Row():
                with gr.Column(scale=1):
                    # Model loading
                    with gr.Group():
                        gr.Markdown("### Model")
                        model_status = gr.Textbox(
                            label="Status",
                            value="Model not loaded",
                            interactive=False,
                        )
                        load_btn = gr.Button("Load Model", variant="primary")

                    # Image input
                    with gr.Group():
                        gr.Markdown("### Input")
                        image_input = gr.Image(
                            label="Upload Image",
                            type="pil",
                        )

                    # Task selection
                    with gr.Group():
                        gr.Markdown("### Task Settings")
                        task_dropdown = gr.Dropdown(
                            choices=[t.value for t in [TaskType.OCR, TaskType.LAYOUT]],
                            value=TaskType.OCR.value,
                            label="Task Type",
                        )
                        prompt_input = gr.Textbox(
                            label="Custom Prompt (optional)",
                            placeholder="Enter custom prompt or leave empty for default",
                            lines=2,
                        )
                        with_boxes = gr.Checkbox(
                            label="Extract with bounding boxes",
                            value=False,
                        )

                    # Generation settings
                    with gr.Accordion("Advanced Settings", open=False):
                        max_tokens = gr.Slider(
                            minimum=100,
                            maximum=8192,
                            value=4096,
                            step=100,
                            label="Max Tokens",
                        )
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                        )

                    # Process button
                    process_btn = gr.Button("Process", variant="primary", size="lg")

                with gr.Column(scale=1):
                    # Output
                    gr.Markdown("### Output")
                    text_output = gr.Textbox(
                        label="Extracted Text",
                        lines=15,
                        show_copy_button=True,
                    )
                    image_output = gr.Image(
                        label="Visualization",
                        type="pil",
                    )

            # Event handlers
            load_btn.click(
                fn=self.load_model,
                outputs=model_status,
            )

            process_btn.click(
                fn=self.process_image,
                inputs=[
                    image_input,
                    task_dropdown,
                    prompt_input,
                    with_boxes,
                    max_tokens,
                    temperature,
                ],
                outputs=[text_output, image_output],
            )

            # Examples
            gr.Markdown("### Quick Start")
            gr.Markdown(
                "1. Click 'Load Model' to initialize the model\n"
                "2. Upload an image\n"
                "3. Select a task (OCR or Layout)\n"
                "4. Click 'Process' to extract information"
            )

        return demo

    def launch(self, **kwargs):
        """Launch the Gradio app."""
        demo = self.create_interface()

        # Default launch settings
        launch_kwargs = {
            "server_name": self.config.server.host,
            "server_port": self.config.server.port,
            "share": self.config.server.share,
        }
        launch_kwargs.update(kwargs)

        demo.launch(**launch_kwargs)


def create_app() -> GradioApp:
    """Create a GradioApp instance."""
    return GradioApp()


def launch_app(**kwargs):
    """Create and launch the Gradio app."""
    app = create_app()
    app.launch(**kwargs)


if __name__ == "__main__":
    launch_app()
