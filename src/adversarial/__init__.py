"""
Adversarial Attacks Module for IoT Device Identification
"""

from .attacks import (
    SensitivityAnalysis,
    AdversarialSearch,
    AdversarialEvaluator,
)
from .trades import (
    TRADESTrainer,
    TRADESAttack,
    create_trades_projection_fn,
)

__all__ = [
    "SensitivityAnalysis",
    "AdversarialSearch",
    "AdversarialEvaluator",
    "TRADESTrainer",
    "TRADESAttack",
    "create_trades_projection_fn",
]
