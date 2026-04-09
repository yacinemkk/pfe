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
    FeatureAttackGenerator,
    MultiAttackTRADES,
    create_trades_projection_fn,
)
from .input_transform import (
    InputTransform,
    MixupTransform,
    AdversarialMixup,
    create_input_transform,
)
from .cutmix import (
    CutMix,
    AdversarialCutMix,
    FeatureCutMix,
    HybridCutMix,
    create_adversarial_cutmix,
)
from .feature_dropout import (
    AdversarialFeatureDropout,
    GradientFeatureMasking,
    FeatureImportanceRegularizer,
    create_adversarial_feature_dropout,
)
from .denoising_autoencoder import (
    DenoisingAutoencoder,
    PerturbationGenerator,
    DenoisingAETrainer,
    create_denoising_autoencoder,
)
from .randomized_smoothing import (
    RandomizedSmoothing,
    AdaptiveRandomizedSmoothing,
    create_randomized_smoothing,
)
from .ensemble import (
    HeterogeneousEnsemble,
    create_heterogeneous_ensemble,
)
from .ibp import (
    IntervalBoundPropagation,
    IBPTrainer,
)

__all__ = [
    "SensitivityAnalysis",
    "AdversarialSearch",
    "AdversarialEvaluator",
    "TRADESTrainer",
    "TRADESAttack",
    "FeatureAttackGenerator",
    "MultiAttackTRADES",
    "create_trades_projection_fn",
    "InputTransform",
    "MixupTransform",
    "AdversarialMixup",
    "create_input_transform",
    "CutMix",
    "AdversarialCutMix",
    "FeatureCutMix",
    "HybridCutMix",
    "create_adversarial_cutmix",
    "AdversarialFeatureDropout",
    "GradientFeatureMasking",
    "FeatureImportanceRegularizer",
    "create_adversarial_feature_dropout",
    "DenoisingAutoencoder",
    "PerturbationGenerator",
    "DenoisingAETrainer",
    "create_denoising_autoencoder",
    "RandomizedSmoothing",
    "AdaptiveRandomizedSmoothing",
    "create_randomized_smoothing",
    "HeterogeneousEnsemble",
    "create_heterogeneous_ensemble",
    "IntervalBoundPropagation",
    "IBPTrainer",
]
