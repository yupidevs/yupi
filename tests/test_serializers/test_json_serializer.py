import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_serializers.utils import compare_trajectories, trajectories
from yupi.core.serializers.json_serializer import JSONSerializer
from yupi.exceptions import LoadTrajectoryError
from yupi.trajectory import Trajectory


@pytest.fixture
def traj_1() -> Trajectory:
    return Trajectory(points=[[1, 4], [2, 5], [3, 6]])


@pytest.fixture
def traj_2() -> Trajectory:
    return Trajectory(points=[[1, 4], [2, 5], [3, 6]], t=[0.0, 0.5, 2.0])


@pytest.fixture
def trajs() -> list[Trajectory]:
    return trajectories()


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


def test_missing_data_on_json(traj_1: Trajectory, traj_2: Trajectory) -> None:
    _path_1 = Path("t1.json")
    _path_2 = Path("t2.json")

    JSONSerializer.save(traj_1, _path_1, overwrite=True)
    JSONSerializer.save(traj_2, _path_2, overwrite=True)

    _missing_axes = Path("missing_axes.json")
    _missing_dt = Path("missing_dt.json")
    _missing_t = Path("missing_t.json")
    _missing_diff_est = Path("missing_diff_est.json")

    missing_axes, missing_dt, missing_t, missing_diff_est = {}, {}, {}, {}
    with _path_1.open("r") as file:
        data = json.load(file)
        missing_axes = data.copy()
        missing_dt = data.copy()
        missing_diff_est = data.copy()

        del missing_axes["axes"]
        del missing_dt["dt"]
        del missing_diff_est["diff_est"]

    with _missing_axes.open("w") as file:
        json.dump(missing_axes, file)

    with _missing_dt.open("w") as file:
        json.dump(missing_dt, file)

    with _missing_diff_est.open("w") as file:
        json.dump(missing_diff_est, file)

    with _path_2.open("r") as file:
        data = json.load(file)
        missing_t = data.copy()
        del missing_t["t"]

    with _missing_t.open("w") as file:
        json.dump(missing_t, file)

    with pytest.raises(LoadTrajectoryError):
        JSONSerializer.load(_missing_axes)

    with pytest.raises(LoadTrajectoryError):
        JSONSerializer.load(_missing_dt)

    with pytest.raises(LoadTrajectoryError):
        JSONSerializer.load(_missing_t)

    # This should not raise an error, if no diff_est present, then default is used
    JSONSerializer.load(_missing_diff_est)

    Path.unlink(_path_1)
    Path.unlink(_path_2)
    Path.unlink(_missing_axes)
    Path.unlink(_missing_dt)
    Path.unlink(_missing_t)
    Path.unlink(_missing_diff_est)


def test_missing_data_on_json_ensemble(trajs: list[Trajectory]) -> None:
    _path = Path("ensemble.json")

    JSONSerializer.save_ensemble(trajs, _path, overwrite=True)

    _missing_axes_enemble = Path("missing_axes_ensemble.json")
    _missing_dt_ensemble = Path("missing_dt_ensemble.json")
    _missing_t_ensemble = Path("missing_t_ensemble.json")

    missing_axes, missing_dt, missing_t = [], [], []
    with _path.open("r") as file:
        data: list[Any] = json.load(file)
        missing_axes = [traj_data.copy() for traj_data in data]
        missing_dt = [traj_data.copy() for traj_data in data]
        missing_t = [traj_data.copy() for traj_data in data]

        del missing_axes[0]["axes"]
        del missing_dt[1]["dt"]
        del missing_t[2]["t"]

    with _missing_axes_enemble.open("w") as file:
        json.dump(missing_axes, file)

    with _missing_dt_ensemble.open("w") as file:
        json.dump(missing_dt, file)

    with _missing_t_ensemble.open("w") as file:
        json.dump(missing_t, file)

    with pytest.raises(LoadTrajectoryError, match="No position data found."):
        JSONSerializer.load_ensemble(_missing_axes_enemble)

    with pytest.raises(LoadTrajectoryError, match="No time data found."):
        JSONSerializer.load_ensemble(_missing_dt_ensemble)

    with pytest.raises(LoadTrajectoryError, match="No time data found."):
        JSONSerializer.load_ensemble(_missing_t_ensemble)

    Path.unlink(_path)
    Path.unlink(_missing_axes_enemble)
    Path.unlink(_missing_dt_ensemble)
    Path.unlink(_missing_t_ensemble)
