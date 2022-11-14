import numpy as np

from yupi.core.featurizers.featurizer import (
    DEFAULT_ZERO_THRESHOLD,
    CompoundFeaturizer,
    GlobalStatsFeaturizer,
)
from yupi.trajectory import Trajectory


class AccelerationGlobalFeaturizer(GlobalStatsFeaturizer):
    """
    Featurizer that computes all the gloabl features related to
    the acceleration of the trajectory.
    """

    def _values(self, traj: Trajectory) -> np.ndarray:
        acc = traj.a.norm
        assert isinstance(acc, np.ndarray)
        return acc


class AccelerationChangeRateGlobalFeaturizer(GlobalStatsFeaturizer):
    """
    Featurizer that computes all the global features related to
    the acceleration change rate of the trajectory.
    """

    def _values(self, traj: Trajectory) -> np.ndarray:
        acc = traj.a.norm
        dt_vals = traj.t.delta
        acc_change_rate = np.diff(acc) / dt_vals
        return acc_change_rate


class AccelerationFeaturizer(CompoundFeaturizer):
    """
    Compound featurizer that computes all the features related to
    the acceleration of the trajectory.
    """

    def __init__(self, zero_threshold: float = DEFAULT_ZERO_THRESHOLD):
        super().__init__(
            AccelerationGlobalFeaturizer(zero_threshold=zero_threshold),
            AccelerationChangeRateGlobalFeaturizer(zero_threshold=zero_threshold),
        )
