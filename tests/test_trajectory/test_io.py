import os

import numpy as np
import pytest

from yupi import Trajectory
from yupi.core.serializers import JSONSerializer, CSVSerializer

APPROX_REL_TOLERANCE = 1e-10


@pytest.fixture
def traj_1() -> Trajectory:
    return Trajectory(points=[[1, 4], [2, 5], [3, 6]])


def trajectories():
    t_1 = Trajectory(points=[[1, 4], [2, 5], [3, 6]])
    t_2 = Trajectory(points=[[1, 4], [2, 5], [3, 6]], t=[0.0, 0.5, 2.0])
    return [t_1, t_2]


def compare_trajectories(t1, t2):
    pytest.approx(t1.t, t2.t, APPROX_REL_TOLERANCE)
    pytest.approx(t1.r, t2.r, APPROX_REL_TOLERANCE)


# Old IO methods
def test_invalid_file_type(traj_1):
    with pytest.raises(ValueError, match="Invalid export file type"):
        traj_1.save("t1", file_type="abc")


def test_overwrite(traj_1):
    traj_1.save("t1")
    traj_1.save("t1")
    with pytest.raises(FileExistsError):
        traj_1.save("t1", overwrite=False)
    os.remove("t1.json")


@pytest.mark.parametrize("traj", trajectories())
def test_old_io_json(traj):
    traj.save("_old_traj", file_type="json")
    loaded_traj = Trajectory.load("_old_traj.json")
    compare_trajectories(traj, loaded_traj)
    os.remove("_old_traj.json")


@pytest.mark.parametrize("traj", trajectories())
def test_old_io_csv(traj):
    traj.save("_old_traj", file_type="csv")
    loaded_traj = Trajectory.load("_old_traj.csv")
    compare_trajectories(traj, loaded_traj)
    os.remove("_old_traj.csv")


# Retrocompatibility
@pytest.mark.parametrize("traj", trajectories())
def test_json_retrocompatibility(traj):
    traj.save("_old_traj", file_type="json")
    loaded_traj = JSONSerializer.load("_old_traj.json")
    compare_trajectories(traj, loaded_traj)
    os.remove("_old_traj.json")


@pytest.mark.parametrize("traj", trajectories())
def test_csv_retrocompatibility(traj):
    traj.save("_old_traj", file_type="csv")
    loaded_traj = CSVSerializer.load("_old_traj.csv")
    compare_trajectories(traj, loaded_traj)
    os.remove("_old_traj.csv")


# Serializers
@pytest.mark.parametrize("traj", trajectories())
def test_json_serializer(traj):
    JSONSerializer.save(traj, "t1.json", overwrite=True)

    with pytest.raises(FileExistsError):
        JSONSerializer.save(traj, "t1.json")

    loaded_traj = JSONSerializer.load("t1.json")
    compare_trajectories(traj, loaded_traj)
    os.remove("t1.json")

@pytest.mark.parametrize("traj", trajectories())
def test_csv_serializer(traj):
    CSVSerializer.save(traj, "t1.csv", overwrite=True)

    with pytest.raises(FileExistsError):
        CSVSerializer.save(traj, "t1.csv")

    loaded_traj = CSVSerializer.load("t1.csv")
    compare_trajectories(traj, loaded_traj)
    os.remove("t1.csv")
