"""Model adapters with lazy imports.

Keeping these imports lazy prevents one model family from requiring every
other model family's optional Python package just to import ``adapters``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "SpeechBrainAdapter",
    "ClearVoiceAdapter",
    "ConvTasNetAdapter",
    "DPRNNAdapter",
]


def __getattr__(name: str) -> Any:
    if name == "SpeechBrainAdapter":
        from .speechbrain_adapter import SpeechBrainAdapter

        return SpeechBrainAdapter
    if name == "ClearVoiceAdapter":
        from .clearvoice_adapter import ClearVoiceAdapter

        return ClearVoiceAdapter
    if name == "ConvTasNetAdapter":
        from .convtasnet_adapter import ConvTasNetAdapter

        return ConvTasNetAdapter
    if name == "DPRNNAdapter":
        from .dprnn_adapter import DPRNNAdapter

        return DPRNNAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
