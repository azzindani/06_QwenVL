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

def notebook_display(content, title="Output Preview", is_html=False):
    """
    Robust display for Notebook environments (Colab, Kaggle).
    Supports images, HTML, and text.
    
    Args:
        content: PIL Image, str path, Path object, or HTML string
        title: Title to display above content
        is_html: If True, treat content as raw HTML
    """
    try:
        from IPython import get_ipython
        if not get_ipython():
            return
    except ImportError:
        return

    try:
        from IPython.display import display, Image as IPImage, HTML
        import io
        
        # Add a title/separator
        display(HTML(f"<div style='margin-top: 10px; border-top: 2px solid #ddd; padding-top: 5px;'><strong>[{title}]</strong></div>"))
        
        # Handle HTML
        if is_html:
            display(HTML(f"<div style='border: 1px solid #eee; padding: 10px; margin-bottom: 10px; overflow-x: auto;'>{content}</div>"))
            return

        # Handle Path/String
        if isinstance(content, (str, Path)) and not is_html:
            display(IPImage(filename=str(content), width=600))
            
        # Handle PIL Image
        elif hasattr(content, 'save'):
            # Convert PIL to raw PNG bytes
            b = io.BytesIO()
            content.save(b, format='PNG')
            display(IPImage(data=b.getvalue(), width=600))
        
        # Fallback for text if needed (though print is usually fine)
        elif isinstance(content, str):
            print(content)
            
    except Exception as e:
        print(f"Error in notebook_display: {e}")
