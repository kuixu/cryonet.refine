"""CryoNet.Refine secondary-structure detection.

The protein and nucleic-acid geometry heuristics are inspired by the
cctbx/mmtbx secondary-structure implementation, but this package is a
standalone implementation using only Gemmi and NumPy.
"""

from .detector import detect_secondary_structure
from .models import DetectionResult

__all__ = ["DetectionResult", "detect_secondary_structure"]
