import numpy as np

from yupi.core.featurizers.featurizer import CompoundFeaturizer, GlobalStatsFeaturizer
from yupi.trajectory import Trajectory


class AccelerationGlobalFeaturizer(GlobalStatsFeaturizer):
    """
    Featurizer that computes all the gloabl features related to
    the acceleration of the trajectory.
    """

    def values(self, traj: Trajectory) -> np.ndarray:
        acc = traj.a.delta.norm
        assert isinstance(acc, np.ndarray)
        return acc


class AccelerationChangeRateGlobalFeaturizer(GlobalStatsFeaturizer):
    """
    Featurizer that computes all the global features related to
    the acceleration change rate of the trajectory.
    """

    def values(self, traj: Trajectory) -> np.ndarray:
        acc = traj.a.delta.norm
        dt_vals = traj.t.delta
        acc_change_rate = np.diff(acc) / dt_vals[1:]
        return acc_change_rate


class AccelerationFeaturizer(CompoundFeaturizer):
    """
    Compound featurizer that computes all the features related to
    the acceleration of the trajectory.
    """

    def __init__(self):
        super().__init__(
            AccelerationGlobalFeaturizer(), AccelerationChangeRateGlobalFeaturizer()
        )
