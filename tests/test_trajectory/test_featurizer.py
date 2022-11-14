from typing import List

import numpy as np
import pytest

from yupi import Trajectory
from yupi.core.featurizers import Featurizer


class SimpleSpacialFeaturizer(Featurizer):
    @property
    def count(self) -> int:
        return 2

    def featurize(self, trajs: List[Trajectory]) -> np.ndarray:
        feats = np.empty((len(trajs), self.count))
        for i, traj in enumerate(trajs):
            feats[i, 0] = float(np.sum(traj.r.delta.norm))  # distance
            feats[i, 1] = float((traj.r[-1] - traj.r[0]).norm)  # displacement
        return feats


class ComplexSpacialFeaturizer(Featurizer):
    @property
    def count(self) -> int:
        return 4

    def featurize(self, trajs: List[Trajectory]) -> np.ndarray:
        feats = np.empty((len(trajs), self.count))
        for i, traj in enumerate(trajs):
            feats[i, 0] = float(np.mean(traj.r.delta.norm))  # jump distance average
            feats[i, 1] = float(np.std(traj.r.delta.norm))  # jump distance std
            feats[i, 2] = float(np.mean(traj.t.delta))  # jump time average
            feats[i, 3] = float(np.std(traj.t.delta))  # jump time std
        return feats


@pytest.fixture
def trajs():
    t1 = Trajectory(x=[0, 0, 4], y=[0, 3, 3])
    t2 = Trajectory(x=[4, 7, 7], y=[4, 4, 8], t=[0, 0.1, 0.2])
    return [t1, t2]


@pytest.fixture
def simple_featurizer():
    return SimpleSpacialFeaturizer()


@pytest.fixture
def complex_featurizer():
    return ComplexSpacialFeaturizer()


def test_featurizer_count(simple_featurizer):
    assert simple_featurizer.count == 2


def test_simple_featurizer(trajs, simple_featurizer):
    feats = simple_featurizer.featurize(trajs)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (2, 2)
    assert pytest.approx(feats) == [[7, 5], [7, 5]]


def test_complex_featurizer(trajs, complex_featurizer):
    feats = complex_featurizer.featurize(trajs)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (2, 4)
    assert pytest.approx(feats) == [
        [3.5, 0.5, 1, 0],
        [3.5, 0.5, 0.1, 0],
    ]


def test_compound_featurizer(trajs, simple_featurizer, complex_featurizer):
    compound = simple_featurizer + complex_featurizer
    feats = compound.featurize(trajs)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (2, 6)
    assert pytest.approx(feats) == [
        [7, 5, 3.5, 0.5, 1, 0],
        [7, 5, 3.5, 0.5, 0.1, 0],
    ]
