from typing import List

import numpy as np

from yupi.core.featurizers.featurizer import (
    DEFAULT_ZERO_THRESHOLD,
    CompoundFeaturizer,
    Featurizer,
    GlobalStatsFeaturizer,
)
from yupi.trajectory import Trajectory


class VelocityGlobalFeaturizer(GlobalStatsFeaturizer):
    """
    Featurizer that computes all the global features related to
    the velocity of the trajectory.
    """

    def _values(self, traj: Trajectory) -> np.ndarray:
        vel = traj.v.norm
        assert isinstance(vel, np.ndarray)
        return vel


class VelocityStopRateFeaturizer(Featurizer):
    """
    Featurizer that computes the rate of stops of the trajectory.

    Parameters
    ----------
    threshold
        The threshold below which the velocity is considered stopped.
    """

    def __init__(self, threshold: float = 1):
        self.threshold = threshold

    @property
    def count(self) -> int:
        return 1

    def featurize(self, trajs: List[Trajectory]) -> np.ndarray:
        feats = np.empty((len(trajs), self.count))
        for i, traj in enumerate(trajs):
            vel = traj.v.norm
            distance = float(np.sum(traj.r.delta.norm))
            assert isinstance(vel, np.ndarray)
            feats[i, 0] = np.sum(vel < self.threshold) / distance
        return feats


class VelocityChangeRateFeaturizer(Featurizer):
    """
    Featurizer that computes the rate of changes of the velocity
    of the trajectory.

    Parameters
    ----------
    threshold : float
        The threshold to consider a change of velocity.
    """

    def __init__(self, threshold: float = 1):
        super().__init__()
        self.threshold = threshold

    @property
    def count(self) -> int:
        return 1

    def featurize(self, trajs: List[Trajectory]) -> np.ndarray:
        feats = np.empty((len(trajs), self.count))
        for i, traj in enumerate(trajs):
            vel = traj.v.norm
            distance = float(np.sum(traj.r.delta.norm))
            assert isinstance(vel, np.ndarray)
            subs = np.abs(np.diff(vel))
            vel[vel == 0] = np.inf
            v_rate = subs / vel[:-1]
            feats[i, 0] = np.sum(v_rate > self.threshold) / distance
        return feats


class VelocityFeaturizer(CompoundFeaturizer):
    """
    Compound featurizer that computes all the features related
    to the velocity of the trajectory.

    Parameters
    ----------
    stop_rate_threshold : float
        The threshold to consider a stop.
    change_rate_threshold : float
        The threshold to consider a change.
    """

    def __init__(
        self,
        stop_rate_threshold: float = 1,
        change_rate_threshold: float = 1,
        zero_threshold: float = DEFAULT_ZERO_THRESHOLD,
    ):
        super().__init__(
            VelocityGlobalFeaturizer(zero_threshold=zero_threshold),
            VelocityStopRateFeaturizer(stop_rate_threshold),
            VelocityChangeRateFeaturizer(change_rate_threshold),
        )
