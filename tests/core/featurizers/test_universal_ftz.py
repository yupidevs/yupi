import pytest

from yupi.core.featurizers import UniversalFeaturizer
from yupi.trajectory import Trajectory


@pytest.fixture
def trajs() -> list[Trajectory]:
    # Create a list of sample trajectories for testing
    t1 = Trajectory(x=[0, 0, 4, 6, 9, 10], y=[0, 3, 3, 4, 5, 6])
    t2 = Trajectory(x=[4, 7, 7, 8, 9, 10], y=[4, 4, 8, 9, 10, 11])
    t3 = Trajectory(x=[1, 2, 3, 4, 5, 6], y=[2, 3, 6, 7, 8, 9])
    return [t1, t2, t3]


def test_universal_feat(trajs: list[Trajectory]) -> None:
    features = UniversalFeaturizer().featurize(trajs)

    # Check that universal contains all features
    assert features.shape == (3, 103)
