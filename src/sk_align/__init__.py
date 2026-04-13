"""sk-align — standalone forced alignment (no Kaldi/PyKaldi dependency)."""

from sk_align.aligner import Aligner

__all__ = ["Aligner"]
__version__ = "0.1.0"


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False
