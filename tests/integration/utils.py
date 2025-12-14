from pathlib import Path
from PIL import Image

def notebook_display(image, title="Image Preview"):
    """
    Robust image display for Notebook environments (Colab, Kaggle).
    Uses raw PNG data to ensure inline rendering.
    
    Args:
        image: PIL Image, str path, or Path object
        title: Title to display above image
    """
    try:
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
