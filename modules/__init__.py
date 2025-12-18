"""
FMCaps Modules
- SGFR: SAM and Grounding-DINO Fusion Refinement
- SGAE: SAM-Guided Affinity Enhancement
"""

from .sgfr import SGFR
from .sgae import SGAE

__all__ = ['SGFR', 'SGAE']

