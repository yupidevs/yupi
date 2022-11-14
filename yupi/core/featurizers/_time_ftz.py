import numpy as np

from yupi.core.featurizers.featurizer import (
    DEFAULT_ZERO_THRESHOLD,
    CompoundFeaturizer,
    GlobalStatsFeaturizer,
)
from yupi.trajectory import Trajectory


class TimeGlobalFeaturizer(GlobalStatsFeaturizer):
    """
    Featurizer that extracts all the gloabl features related
    to the time array of the trajectories

    Parameters
    ----------
    from_zero : bool, optional
        If True, the time array is shifted to start from
        zero, by default True.
    """

    def __init__(
        self, from_zero: bool = True, zero_threshold: float = DEFAULT_ZERO_THRESHOLD
    ):
        super().__init__(zero_threshold=zero_threshold)
        self.from_zero = from_zero

    def _values(self, traj: Trajectory) -> np.ndarray:
        time: np.ndarray = traj.t
        if self.from_zero:
            time -= time[0]
        return time


class TimeJumpsGlobalFeaturizer(GlobalStatsFeaturizer):
    """
    Featurizer that extracts all the gloabl features related
    to the time intervals of the trajectories
    """

    def _values(self, traj: Trajectory) -> np.ndarray:
        return traj.t.delta


class TimeFeaturizer(CompoundFeaturizer):
    """
    Compound featurizer that extracts all the features related
    to the time array of the trajectories
    """

    def __init__(
        self, from_zero: bool = True, zero_threshold: float = DEFAULT_ZERO_THRESHOLD
    ):
        super().__init__(
            TimeGlobalFeaturizer(from_zero=from_zero, zero_threshold=zero_threshold),
            TimeJumpsGlobalFeaturizer(zero_threshold=zero_threshold),
        )
