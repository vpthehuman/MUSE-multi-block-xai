# src/muse_xai/__init__.py
"""Top-level package for muse_xai."""

# Re-export the main API expected by tests:
from .core import MUSE  # noqa: F401

__all__ = ["MUSE"]
