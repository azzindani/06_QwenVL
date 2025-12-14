"""Video understanding task handler."""

import hashlib
import logging
import math
import os
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from .base import BaseTaskHandler, TaskResult, TaskType, register_handler

logger = logging.getLogger(__name__)


def get_video_frames(
    video_path: str,
    num_frames: int = 32,
    cache_dir: str = ".cache",
) -> tuple:
    """
    Extract frames from a video with caching.
    
    Args:
        video_path: Path to video file or URL
        num_frames: Number of frames to extract
        cache_dir: Directory for caching frames
        
    Returns:
        Tuple of (video_path, frames, timestamps)
    """
    try:
        import numpy as np
        from decord import VideoReader, cpu
    except ImportError:
        raise ImportError(
            "Video processing requires 'decord' library. "
            "Install with: pip install decord"
        )

    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode("utf-8")).hexdigest()
    
    # Handle URL videos
    if video_path.startswith("http://") or video_path.startswith("https://"):
        video_file_path = os.path.join(cache_dir, f"{video_hash}.mp4")
        if not os.path.exists(video_file_path):
            import requests
            response = requests.get(video_path)
            with open(video_file_path, "wb") as f:
                f.write(response.content)
    else:
        video_file_path = video_path

    # Check cache
    frames_cache_file = os.path.join(cache_dir, f"{video_hash}_{num_frames}_frames.npy")
    timestamps_cache_file = os.path.join(cache_dir, f"{video_hash}_{num_frames}_timestamps.npy")

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    # Extract frames
    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    # Cache frames
    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)

    return video_file_path, frames, timestamps


def create_image_grid(images: list, num_columns: int = 8) -> Image.Image:
    """
    Create a grid image from a list of frames.
    
    Args:
        images: List of numpy arrays (frames)
        num_columns: Number of columns in the grid
        
    Returns:
        PIL Image containing the grid
    """
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = math.ceil(len(images) / num_columns)

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new("RGB", (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image


@register_handler(TaskType.VIDEO)
class VideoHandler(BaseTaskHandler):
    """Handler for video understanding tasks."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.VIDEO

    @property
    def system_prompt(self) -> str:
        return (
            "You are a helpful assistant specialized in video understanding and analysis. "
            "Analyze video content to describe actions, events, objects, and temporal relationships. "
            "Provide detailed and accurate descriptions of what happens in the video."
        )

    def process(
        self,
        video: str,
        prompt: Optional[str] = None,
        num_frames: int = 32,
        **kwargs,
    ) -> TaskResult:
        """
        Analyze a video.

        Args:
            video: Path to video file
            prompt: Custom prompt for analysis
            num_frames: Number of frames to sample

        Returns:
            TaskResult with video analysis
        """
        user_prompt = prompt or "Describe what happens in this video in detail."

        # Build video message
        messages = self._build_video_messages(video, user_prompt, num_frames)
        response = self._generate_video(messages, **kwargs)

        return TaskResult(
            text=response,
            metadata={
                "mode": "video_understanding",
                "num_frames": num_frames,
            },
        )

    def _build_video_messages(
        self,
        video_path: str,
        user_prompt: str,
        num_frames: int = 32,
    ) -> List[Dict[str, Any]]:
        """Build chat messages for video input."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "total_pixels": 20480 * 28 * 28,
                        "min_pixels": 16 * 28 * 28,
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        return messages

    def _generate_video(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate response for video input."""
        import torch

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError(
                "Video processing requires 'qwen-vl-utils' library. "
                "Install with: pip install qwen-vl-utils"
            )

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process video
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages], return_video_kwargs=True
        )
        fps_inputs = video_kwargs.get("fps", [])

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=fps_inputs if fps_inputs else None,
            padding=True,
            return_tensors="pt",
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

    def summarize(
        self,
        video: str,
        num_frames: int = 32,
        **kwargs,
    ) -> TaskResult:
        """
        Summarize video content.

        Args:
            video: Path to video file
            num_frames: Number of frames to sample

        Returns:
            TaskResult with video summary
        """
        prompt = (
            "Provide a concise summary of this video. "
            "Include the main events, actions, and any important details."
        )

        return self.process(video, prompt=prompt, num_frames=num_frames, **kwargs)

    def extract_key_moments(
        self,
        video: str,
        num_frames: int = 32,
        **kwargs,
    ) -> TaskResult:
        """
        Extract key moments from video.

        Args:
            video: Path to video file
            num_frames: Number of frames to sample

        Returns:
            TaskResult with key moments
        """
        prompt = (
            "Identify the key moments in this video. "
            "For each key moment, describe what happens and when it occurs "
            "(early, middle, or late in the video)."
        )

        return self.process(video, prompt=prompt, num_frames=num_frames, **kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("VIDEO HANDLER TEST")
    print("=" * 60)
    print("  Video handler registered successfully")
    print(f"  Task type: {TaskType.VIDEO}")
    print("  Note: Requires 'decord' library for video frame extraction")
    print("  (Actual inference requires loaded model)")
    print("=" * 60)
