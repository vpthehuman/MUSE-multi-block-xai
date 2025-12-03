# expose the main API symbol expected by tests
from .core import MUSE  # noqa: F401

__all__ = ["MUSE"]
__version__ = "0.1.0"
git add src/muse_xai/__init__.py
git commit -m "Bump package version to 0.1.0"
git tag -a v0.1.0 -m "v0.1.0 release"
git push origin main --tags
