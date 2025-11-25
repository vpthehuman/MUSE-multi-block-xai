"""
MUSE – Multi-block Utility for Safe & Explainable learning.

High-level entry point: the `MUSE` classifier for multi-block tabular data
with SHAP-based global/local explanations and model cards.
"""

from .core import MUSE

__all__ = ["MUSE"]
__version__ = "0.1.0"

