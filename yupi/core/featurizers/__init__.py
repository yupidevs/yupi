"""
A Featurizer is a structure that takes a set of trajectories and returns a
feature matrix. The feature matrix is a 2D numpy array with shape (n_trajs,
n_features).

This module contains the basic structures for creating featurizers along
with the implementation of some of them which compute the most common
features used in the literature.
"""

from yupi.core.featurizers._acceleration_ftz import (
    AccelerationChangeRateGlobalFeaturizer,
    AccelerationFeaturizer,
    AccelerationGlobalFeaturizer,
)
from yupi.core.featurizers._angle_ftz import (
    AngleFeaturizer,
    AngleGlobalFeaturizer,
    TurningAngleChangeRateGlobalFeaturizer,
    TurningAngleGobalFeaturizer,
)
from yupi.core.featurizers._kinetic_ftz import KineticFeaturizer
from yupi.core.featurizers._spatial_ftz import (
    DisplacementFeaturizer,
    DistanceFeaturizer,
    SpatialFeaturizer,
)
from yupi.core.featurizers._time_ftz import (
    TimeFeaturizer,
    TimeGlobalFeaturizer,
    TimeJumpsGlobalFeaturizer,
)
from yupi.core.featurizers._universal_ftz import UniversalFeaturizer
from yupi.core.featurizers._velocity_ftz import (
    VelocityChangeRateFeaturizer,
    VelocityFeaturizer,
    VelocityGlobalFeaturizer,
    VelocityStopRateFeaturizer,
)
from yupi.core.featurizers.featurizer import (
    DEFAULT_ZERO_THRESHOLD,
    CompoundFeaturizer,
    Featurizer,
    GlobalStatsFeaturizer,
)

__all__ = [
    "AccelerationChangeRateGlobalFeaturizer",
    "AccelerationFeaturizer",
    "AccelerationGlobalFeaturizer",
    "AngleFeaturizer",
    "AngleGlobalFeaturizer",
    "CompoundFeaturizer",
    "DEFAULT_ZERO_THRESHOLD",
    "DisplacementFeaturizer",
    "DistanceFeaturizer",
    "Featurizer",
    "GlobalStatsFeaturizer",
    "KineticFeaturizer",
    "SpatialFeaturizer",
    "TimeFeaturizer",
    "TimeGlobalFeaturizer",
    "TimeJumpsGlobalFeaturizer",
    "TurningAngleChangeRateGlobalFeaturizer",
    "TurningAngleGobalFeaturizer",
    "UniversalFeaturizer",
    "VelocityChangeRateFeaturizer",
    "VelocityFeaturizer",
    "VelocityGlobalFeaturizer",
    "VelocityStopRateFeaturizer",
]
