"""
JSON trajctory serializer.
"""

import json
import logging
from typing import List

import yupi._differentiation as diff
from yupi import Trajectory
from yupi.core.serializers.serializer import Serializer
from yupi.exceptions import LoadTrajectoryError


class JSONSerializer(Serializer):
    """
    Handles trajectory files in JSON format.
    """

    @staticmethod
    def save(
        traj: Trajectory, file_name: str, overwrite: bool = False, **kwargs
    ) -> None:
        """
        Writes a trajectory to a file.

        Parameters
        ----------
        traj : Trajectory
            The trajectory to write to the file.
        file_name : str
            The name of the file to write.
        overwrite : bool
            If True, overwrites the file if it already exists.
        kwargs
            Additional arguments to pass to the ``open`` function.

            Encoding is set to UTF-8 as default.
        """
        JSONSerializer.check_save_path(
            file_name, overwrite=overwrite, extension=".json"
        )

        json_dict = JSONSerializer.to_json(traj)
        encoding = "utf-8" if "encoding" not in kwargs else kwargs.pop("encoding")
        with open(file_name, "w", encoding=encoding, **kwargs) as traj_file:
            json.dump(json_dict, traj_file)

    @staticmethod
    def save_ensemble(
        trajs: List[Trajectory], file_name: str, overwrite: bool = False, **kwargs
    ) -> None:
        """
        Writes an ensemble to a file.

        The main difference with the ``save`` method is that all the
        trajectories are saved in a single file.

        Parameters
        ----------
        trajs : List[Trajectory]
            The ensemble to write to the file.
        file_name : str
            The name of the file to write.
        overwrite : bool
            If True, overwrites the file if it already exists.
        kwargs
            Additional arguments to pass to the ``open`` function.

            Encoding is set to UTF-8 as default.
        """
        JSONSerializer.check_save_path(
            file_name, overwrite=overwrite, extension=".json"
        )

        json_dicts = [JSONSerializer.to_json(traj) for traj in trajs]
        encoding = "utf-8" if "encoding" not in kwargs else kwargs.pop("encoding")
        with open(file_name, "w", encoding=encoding, **kwargs) as traj_file:
            json.dump(json_dicts, traj_file)

    @staticmethod
    def load(file_name: str, **kwargs) -> Trajectory:
        """
        Loads a trajectory from a file.

        Parameters
        ----------
        file_name : str
            The name of the file to loaded.
        kwargs : dict
            Additional keyword arguments.

            Encoding is set to UTF-8 as default.

        Returns
        -------
        Trajectory
            The trajectory loaded from the file.
        """
        JSONSerializer.check_load_path(file_name, extension=".json")

        encoding = "utf-8" if "encoding" not in kwargs else kwargs.pop("encoding")
        with open(file_name, "r", encoding=encoding, **kwargs) as file:
            data = json.load(file)

            if "axes" not in data and "r" not in data:
                raise LoadTrajectoryError(file_name, "No position data found.")
            if "dt" not in data and "t" not in data:
                raise LoadTrajectoryError(file_name, "No time data found.")
            return JSONSerializer.from_json(data)

    @staticmethod
    def load_ensemble(file_name: str, **kwargs) -> List[Trajectory]:
        """
        Loads an ensemble from a file.

        The main difference with the ``load`` method is that all the
        trajectories are loaded from a single file.

        Parameters
        ----------
        file_name : str
            The name of the file to loaded.
        kwargs : dict
            Additional keyword arguments.

            Encoding is set to UTF-8 as default.

        Returns
        -------
        List[Trajectory]
            The ensemble loaded from the file.
        """
        JSONSerializer.check_load_path(file_name, extension=".json")

        encoding = "utf-8" if "encoding" not in kwargs else kwargs.pop("encoding")
        with open(file_name, "r", encoding=encoding, **kwargs) as file:
            data = json.load(file)

            if any("axes" not in traj and "r" not in traj for traj in data):
                raise LoadTrajectoryError(
                    file_name, "No position data found for one or more trajectories."
                )
            if any("dt" not in traj and "t" not in traj for traj in data):
                raise LoadTrajectoryError(
                    file_name, "No time data found for one or more trajectories."
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
        window = Trajectory.general_diff_est.get("window_type", diff.WindowType.CENTRAL)
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
        axes = json_traj.get("axes", None)
        if axes is None:
            logging.warning(
                "Trajectory will be loaded but it seems to be saved in an old format. "
                "Please consider updating it by using the JSONSerializer.save method. "
                "Older format won't be supported in a future."
            )
            axes = list(json_traj["r"].values())
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

        return Trajectory(
            axes=axes, t=t, dt=dt, t_0=t_0, traj_id=traj_id, diff_est=diff_est
        )
