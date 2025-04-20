# Import PyTorch models by default
try:
    from .pytorch_model import PreferenceModelPyTorch as PreferenceModel
    from .pytorch_model import EnsembleModelPyTorch as EnsembleModel
    USE_PYTORCH = True
except ImportError:
    # Fall back to TensorFlow models if PyTorch isn't available
    from .model import PreferenceModel
    from .model import EnsembleModel
    USE_PYTORCH = False

from .data_collector import DataCollector
from .image_processor import ImageProcessor
from .tinder_client import TinderClient
from .preference_recognizer import PreferenceRecognizer

# Print a message to indicate which backend is being used
import sys
print(f"Using {'PyTorch' if USE_PYTORCH else 'TensorFlow'} backend for models")

__all__ = [
    'PreferenceModel',
    'EnsembleModel',
    'DataCollector',
    'ImageProcessor',
    'TinderClient',
    'PreferenceRecognizer',
    'USE_PYTORCH'
]