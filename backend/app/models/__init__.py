"""Leak-detection models. Register new models in REGISTRY; the UI lists them by name."""

from .base import LeakModel, ModelInput, ModelOutput, DetectionEvent
from .baseline import ResidualBaseline
from .lstm_gnn import LstmGnnModel

REGISTRY: dict[str, LeakModel] = {m.id: m for m in [LstmGnnModel(), ResidualBaseline()]}

__all__ = ["LeakModel", "ModelInput", "ModelOutput", "DetectionEvent", "REGISTRY"]
