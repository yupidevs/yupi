from typing import List

import numpy as np

from yupi.core.featurizers.featurizer import (
    DEFAULT_ZERO_THRESHOLD,
    CompoundFeaturizer,
    Featurizer,
    GlobalStatsFeaturizer,
)
from yupi.trajectory import Trajectory


class DistanceFeaturizer(Featurizer):
    """
    Featurizer that computes the distance of the trajectory.
    """

    @property
    def count(self) -> int:
        return 1

    def featurize(self, trajs: List[Trajectory]) -> np.ndarray:
        feats = np.empty((len(trajs), self.count))
        for i, traj in enumerate(trajs):
            feats[i, :] = float(np.sum(traj.r.delta.norm))
        return feats


class DisplacementFeaturizer(Featurizer):
    """
    Featurizer that computes the displacement of the trajectory.
    """

    @property
    def count(self) -> int:
        return 1

    def featurize(self, trajs: List[Trajectory]) -> np.ndarray:
        feats = np.empty((len(trajs), self.count))
        for i, traj in enumerate(trajs):
            feats[i, :] = float((traj.r[-1] - traj.r[0]).norm)
        return feats


class JumpsGlobalFeaturizer(GlobalStatsFeaturizer):
    """
    Featurizer all the global features related to the jumps
    between each point of the trajectory.
    """

    def _values(self, traj: Trajectory) -> np.ndarray:
        jumps = traj.r.delta.norm
        assert isinstance(jumps, np.ndarray)
        return jumps


class SpatialFeaturizer(CompoundFeaturizer):
    """
    Compound featurizer that computes all the global features related to
    the spatial characteristics of the trajectory.
    """

    def __init__(self, zero_threshold: float = DEFAULT_ZERO_THRESHOLD):
        super().__init__(
            DistanceFeaturizer(zero_threshold=zero_threshold),
            DisplacementFeaturizer(zero_threshold=zero_threshold),
            JumpsGlobalFeaturizer(zero_threshold=zero_threshold),
        )
