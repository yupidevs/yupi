"""
JSON trajctory serializer.
"""

import json
from pathlib import Path
from typing import Any

import yupi._differentiation as diff
from yupi.core.serializers.serializer import Serializer
from yupi.exceptions import LoadTrajectoryError
from yupi.trajectory import Trajectory


class JSONSerializer(Serializer):
    """
    Handles trajectory files in JSON format.
    """

    @staticmethod
    def save(
        traj: Trajectory, file_path: str | Path, overwrite: bool = False, **kwargs: Any
    ) -> None:
        """
        Writes a trajectory to a file.

        Parameters
        ----------
        traj : Trajectory
            The trajectory to write to the file.
        file_path : str | Path
            The path of the file to write.
        overwrite : bool
            If True, overwrites the file if it already exists.
        kwargs
            Additional arguments to pass to the ``open`` function.

            Encoding is set to UTF-8 as default.
        """
        _path = Path(file_path) if isinstance(file_path, str) else file_path

        JSONSerializer.check_save_path(_path, overwrite=overwrite, extension=".json")

        json_dict = JSONSerializer.to_json(traj)
        encoding = "utf-8" if "encoding" not in kwargs else kwargs.pop("encoding")
        with _path.open("w", encoding=encoding, **kwargs) as traj_file:
            json.dump(json_dict, traj_file)

    @staticmethod
    def save_ensemble(
        trajs: list[Trajectory],
        file_path: str | Path,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Writes an ensemble to a file.

        The main difference with the ``save`` method is that all the
        trajectories are saved in a single file.

        Parameters
        ----------
        trajs : list[Trajectory]
            The ensemble to write to the file.
        file_path : str | Path
            The path of the file to write.
        overwrite : bool
            If True, overwrites the file if it already exists.
        kwargs
            Additional arguments to pass to the ``open`` function.

            Encoding is set to UTF-8 as default.
        """
        _path = Path(file_path) if isinstance(file_path, str) else file_path

        JSONSerializer.check_save_path(_path, overwrite=overwrite, extension=".json")

        json_dicts = [JSONSerializer.to_json(traj) for traj in trajs]
        encoding = "utf-8" if "encoding" not in kwargs else kwargs.pop("encoding")
        with _path.open("w", encoding=encoding, **kwargs) as traj_file:
            json.dump(json_dicts, traj_file)

    @staticmethod
    def load(file_path: str | Path, **kwargs: Any) -> Trajectory:
        """
        Loads a trajectory from a file.

        Parameters
        ----------
        file_path : str | Path
            The path of the file to loaded.
        kwargs : dict
            Additional keyword arguments.

            Encoding is set to UTF-8 as default.

        Returns
        -------
        Trajectory
            The trajectory loaded from the file.
        """
        _path = Path(file_path) if isinstance(file_path, str) else file_path

        JSONSerializer.check_load_path(_path, extension=".json")

        encoding = "utf-8" if "encoding" not in kwargs else kwargs.pop("encoding")
        with _path.open("r", encoding=encoding, **kwargs) as file:
            data = json.load(file)

            if "axes" not in data:
                raise LoadTrajectoryError(str(_path), "No position data found.")
            if "dt" not in data and "t" not in data:
                raise LoadTrajectoryError(str(_path), "No time data found.")
            return JSONSerializer.from_json(data)

    @staticmethod
    def load_ensemble(file_path: str | Path, **kwargs: Any) -> list[Trajectory]:
        """
        Loads an ensemble from a file.

        The main difference with the ``load`` method is that all the
        trajectories are loaded from a single file.

        Parameters
        ----------
        file_path : str | Path
            The path of the file to loaded.
        kwargs : dict
            Additional keyword arguments.

            Encoding is set to UTF-8 as default.

        Returns
        -------
        list[Trajectory]
            The ensemble loaded from the file.
        """
        _path = Path(file_path) if isinstance(file_path, str) else file_path

        JSONSerializer.check_load_path(_path, extension=".json")

        encoding = "utf-8" if "encoding" not in kwargs else kwargs.pop("encoding")
        with _path.open("r", encoding=encoding, **kwargs) as file:
            data = json.load(file)

            if any("axes" not in traj for traj in data):
                raise LoadTrajectoryError(
                    str(_path),
                    "No position data found for one or more trajectories.",
                )
            if any("dt" not in traj and "t" not in traj for traj in data):
                raise LoadTrajectoryError(
                    str(_path), "No time data found for one or more trajectories."
                )
            return [JSONSerializer.from_json(traj) for traj in data]

    @staticmethod
    def to_json(traj: Trajectory) -> dict:
        """
        Converts a trajectory to a JSON dictionary.

        Parameters
        ----------
        traj : Trajectory
            The trajectory to convert.

        Returns
        -------
        dict
            The JSON dictionary.
        """

        method = Trajectory.general_diff_est.get("method", diff.DiffMethod.LINEAR_DIFF)
        window = Trajectory.general_diff_est.get("window_type", diff.WindowType.FORWARD)
        accuracy = Trajectory.general_diff_est.get("accuracy", 1)
        diff_est = {
            "method": traj.diff_est.get("method", method).value,
            "window_type": traj.diff_est.get("window_type", window).value,
            "accuracy": traj.diff_est.get("accuracy", accuracy),
        }

        json_dict = {
            "axes": traj.r.T.tolist(),
            "id": traj.traj_id,
            "diff_est": diff_est,
        }
        if traj.dt_std == 0:
            json_dict["dt"] = traj.dt
            json_dict["t_0"] = traj.t_0
        else:
            json_dict["t"] = traj.t.tolist()

        if traj.extra:
            json_dict["extra"] = traj.extra

        if traj.metadata:
            json_dict["metadata"] = traj.metadata

        return json_dict

    @staticmethod
    def from_json(json_traj: dict) -> Trajectory:
        """
        Converts a JSON dictionary to a trajectory.

        Parameters
        ----------
        json_traj : dict
            The JSON dictionary to convert.

        Returns
        -------
        Trajectory
            The trajectory.
        """
        axes = json_traj["axes"]
        traj_id = json_traj["id"] if json_traj["id"] is not None else ""

        diff_est = json_traj.get("diff_est", None)
        if diff_est is None:
            diff_est = Trajectory.general_diff_est
        else:
            diff_est["method"] = diff.DiffMethod(diff_est["method"])
            diff_est["window_type"] = diff.WindowType(diff_est["window_type"])

        t = json_traj.get("t", None)
        dt = json_traj.get("dt", None)
        t_0 = json_traj.get("t_0", 0.0)

        extra = json_traj.get("extra", None)
        metadata = json_traj.get("metadata", {})

        return Trajectory(
            axes=axes,
            extra=extra,
            t=t,
            dt=dt,
            t_0=t_0,
            traj_id=traj_id,
            diff_est=diff_est,
            **metadata,
        )
