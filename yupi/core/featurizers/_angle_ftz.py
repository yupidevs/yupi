import numpy as np

from yupi.core.featurizers.featurizer import (
    DEFAULT_ZERO_THRESHOLD,
    CompoundFeaturizer,
    GlobalStatsFeaturizer,
)
from yupi.stats.turning_angles import turning_angles
from yupi.trajectory import Trajectory


class AngleGlobalFeaturizer(GlobalStatsFeaturizer):
    """
    Featurizer that computes all the global features related to
    the angles of the trajectory.
    """

    def _values(self, traj: Trajectory) -> np.ndarray:
        return turning_angles(traj, accumulate=True)


class TurningAngleGobalFeaturizer(GlobalStatsFeaturizer):
    """
    Featurizer that computes all the global features related to
    the turning angles of the trajectory.
    """

    def _values(self, traj: Trajectory) -> np.ndarray:
        return turning_angles(traj, accumulate=False)


class TurningAngleChangeRateGlobalFeaturizer(GlobalStatsFeaturizer):
    """
    Featurizer that computes all the global features related to
    the turning angle change rate of the trajectory.
    """

    def _values(self, traj: Trajectory) -> np.ndarray:
        angles = turning_angles(traj, accumulate=False)
        dt_vals = traj.t.delta
        angle_change_rate = np.diff(angles) / dt_vals[2:]
        return angle_change_rate


class AngleFeaturizer(CompoundFeaturizer):
    """
    Compound featurizer that computes all the features related to
    the angles of the trajectory.
    """

    def __init__(self, zero_threshold: float = DEFAULT_ZERO_THRESHOLD):
        super().__init__(
            AngleGlobalFeaturizer(zero_threshold=zero_threshold),
            TurningAngleGobalFeaturizer(zero_threshold=zero_threshold),
            TurningAngleChangeRateGlobalFeaturizer(zero_threshold=zero_threshold),
        )
