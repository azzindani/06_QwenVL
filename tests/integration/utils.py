from pathlib import Path
from PIL import Image
import gc
import torch

def cleanup_memory():
    """Force garbage collection and empty CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def is_oom_error(e):
    """Check if exception is a GPU Out Of Memory error."""
    msg = str(e).lower()
    return "out of memory" in msg or "failed to allocate" in msg or "cuda out of memory" in msg

def notebook_display(image, title="Image Preview"):
    """
    Robust image display for Notebook environments (Colab, Kaggle).
    Uses raw PNG data to ensure inline rendering.
    
    Args:
        image: PIL Image, str path, or Path object
        title: Title to display above image
    """
    try:
        # Check environment - avoid printing objects in non-interactive shells
        try:
            from IPython import get_ipython
            if not get_ipython():
                print("  [Info] Run with '%run' or import main() to see inline image previews")
                return
        except ImportError:
            return

        from IPython.display import display, Image as IPImage, HTML
        import io
        
        # Add a title/separator
        display(HTML(f"<strong>[{title}]</strong>"))
        
        # Handle Path/String
        if isinstance(image, (str, Path)):
            display(IPImage(filename=str(image), width=600))
            
        # Handle PIL Image
        elif hasattr(image, 'save'):
            # Convert PIL to raw PNG bytes
            b = io.BytesIO()
            image.save(b, format='PNG')
            display(IPImage(data=b.getvalue(), width=600))
            
    except ImportError:
        pass  # Not in a notebook environment
