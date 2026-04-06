"""
Adversarial Attacks Module for IoT Device Identification
"""

from .attacks import (
    FeatureLevelAttack,
    SequenceLevelAttack,
    HybridAdversarialAttack,
    AdversarialEvaluator,
)
from .trades import (
    TRADESTrainer,
    TRADESAttack,
    create_trades_projection_fn,
)

__all__ = [
    "FeatureLevelAttack",
    "SequenceLevelAttack",
    "HybridAdversarialAttack",
    "AdversarialEvaluator",
    "TRADESTrainer",
    "TRADESAttack",
    "create_trades_projection_fn",
]
