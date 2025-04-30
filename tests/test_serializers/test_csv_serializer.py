from pathlib import Path

import pytest

from tests.test_serializers.utils import compare_trajectories, trajectories
from yupi import Trajectory
from yupi.core.serializers import CSVSerializer


@pytest.mark.parametrize("traj", trajectories())
def test_csv_serializer(traj: Trajectory) -> None:
    _path = Path("t1.csv")
    CSVSerializer.save(traj, _path, overwrite=True)

    with pytest.raises(FileExistsError):
        CSVSerializer.save(traj, _path)

    loaded_traj = CSVSerializer.load(_path)
    compare_trajectories(traj, loaded_traj)
    Path.unlink(_path)
