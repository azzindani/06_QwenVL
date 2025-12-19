import gc
import logging

logger = logging.getLogger(__name__)

def cleanup_memory(force_gc: bool = True, empty_cuda_cache: bool = True):
    """
    Carefully clean up memory by running garbage collection and optionally 
    emptying the CUDA cache.
    
    Args:
        force_gc: Whether to run gc.collect()
        empty_cuda_cache: Whether to run torch.cuda.empty_cache()
    """
    if force_gc:
        # Run multiple collections to handle cyclic references
        for _ in range(2):
            gc.collect()
            
    if empty_cuda_cache:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache emptied")
        except ImportError:
            pass
