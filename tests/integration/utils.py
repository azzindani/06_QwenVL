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

def resize_for_display(img, max_width=1200, max_height=1200):
    """Resize image for display while maintaining aspect ratio."""
    if not hasattr(img, 'size'):
        return img
    
    width, height = img.size
    if width <= max_width and height <= max_height:
        return img
    
    # Calculate scale factor
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

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
            img = Image.open(str(content))
            resized = resize_for_display(img)
            b = io.BytesIO()
            resized.save(b, format='PNG')
            print(f"    Original: {img.size}, Display: {resized.size}")
            display(IPImage(data=b.getvalue()))
            
        # Handle PIL Image
        elif hasattr(content, 'save'):
            # Resize for display
            orig_w, orig_h = content.size
            resized = resize_for_display(content)
            disp_w, disp_h = resized.size
            scale = disp_w / orig_w
            b = io.BytesIO()
            resized.save(b, format='PNG')
            print(f"    Original: {orig_w}x{orig_h}, Display: {disp_w}x{disp_h}, Scale: {scale:.3f}")
            display(IPImage(data=b.getvalue()))
        
        # Fallback for text if needed (though print is usually fine)
        elif isinstance(content, str):
            print(content)
            
    except Exception as e:
        print(f"Error in notebook_display: {e}")
