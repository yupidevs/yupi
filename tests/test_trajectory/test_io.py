from pathlib import Path

import pytest

from yupi import Trajectory
from yupi.core.serializers import CSVSerializer, JSONSerializer

APPROX_REL_TOLERANCE = 1e-10


@pytest.fixture
def traj_1() -> Trajectory:
    return Trajectory(points=[[1, 4], [2, 5], [3, 6]])


def trajectories() -> list[Trajectory]:
    t_1 = Trajectory(points=[[1, 4], [2, 5], [3, 6]])
    t_2 = Trajectory(points=[[1, 4], [2, 5], [3, 6]], t=[0.0, 0.5, 2.0])
    return [t_1, t_2]


def compare_trajectories(t1: Trajectory, t2: Trajectory) -> None:
    pytest.approx(t1.t, t2.t, APPROX_REL_TOLERANCE)
    pytest.approx(t1.r, t2.r, APPROX_REL_TOLERANCE)


# Old IO methods
def test_invalid_file_type(traj_1: Trajectory) -> None:
    with pytest.raises(ValueError, match="Invalid export file type"):
        traj_1.save("t1", file_type="abc")


def test_overwrite(traj_1: Trajectory) -> None:
    traj_1.save("t1")
    traj_1.save("t1")
    with pytest.raises(FileExistsError):
        traj_1.save("t1", overwrite=False)
    Path.unlink(Path("t1.json"))


@pytest.mark.parametrize("traj", trajectories())
def test_old_io_json(traj: Trajectory) -> None:
    traj.save("_old_traj", file_type="json")
    loaded_traj = Trajectory.load("_old_traj.json")
    compare_trajectories(traj, loaded_traj)
    Path.unlink(Path("_old_traj.json"))


@pytest.mark.parametrize("traj", trajectories())
def test_old_io_csv(traj: Trajectory) -> None:
    traj.save("_old_traj", file_type="csv")
    loaded_traj = Trajectory.load("_old_traj.csv")
    compare_trajectories(traj, loaded_traj)
    Path.unlink(Path("_old_traj.csv"))


# Retrocompatibility
@pytest.mark.parametrize("traj", trajectories())
def test_json_retrocompatibility(traj: Trajectory) -> None:
    traj.save("_old_traj", file_type="json")
    loaded_traj = JSONSerializer.load("_old_traj.json")
    compare_trajectories(traj, loaded_traj)
    Path.unlink(Path("_old_traj.json"))


@pytest.mark.parametrize("traj", trajectories())
def test_csv_retrocompatibility(traj: Trajectory) -> None:
    traj.save("_old_traj", file_type="csv")
    loaded_traj = CSVSerializer.load("_old_traj.csv")
    compare_trajectories(traj, loaded_traj)
    Path.unlink(Path("_old_traj.csv"))


# Serializers
@pytest.mark.parametrize("traj", trajectories())
def test_json_serializer(traj: Trajectory) -> None:
    _path = Path("t1.json")
    JSONSerializer.save(traj, _path, overwrite=True)

    with pytest.raises(FileExistsError):
        JSONSerializer.save(traj, _path)

    loaded_traj = JSONSerializer.load("t1.json")
    compare_trajectories(traj, loaded_traj)
    Path.unlink(_path)

    _ensemble_path = Path("ensemble.json")
    ensemble = [traj, traj]
    JSONSerializer.save_ensemble(ensemble, _ensemble_path, overwrite=True)

    with pytest.raises(FileExistsError):
        JSONSerializer.save_ensemble(ensemble, _ensemble_path)

    loaded_trajs = JSONSerializer.load_ensemble(_ensemble_path)
    for t_1, t_2 in zip(ensemble, loaded_trajs, strict=True):
        compare_trajectories(t_1, t_2)
    Path.unlink(_ensemble_path)


@pytest.mark.parametrize("traj", trajectories())
def test_csv_serializer(traj: Trajectory) -> None:
    _path = Path("t1.csv")
    CSVSerializer.save(traj, _path, overwrite=True)

    with pytest.raises(FileExistsError):
        CSVSerializer.save(traj, _path)

    loaded_traj = CSVSerializer.load(_path)
    compare_trajectories(traj, loaded_traj)
    Path.unlink(_path)
