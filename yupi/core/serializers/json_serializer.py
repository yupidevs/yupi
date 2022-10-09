"""
JSON trajctory serializer.
"""

import json
import logging

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
        kwargs["encoding"] = kwargs.get("encoding", "utf-8")
        with open(file_name, "w", **kwargs) as traj_file:
            json.dump(json_dict, traj_file)

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

        kwargs["encoding"] = kwargs.get("encoding", "utf-8")
        with open(file_name, "r", **kwargs) as file:
            data = json.load(file)

            if "axes" not in data and "r" not in data:
                raise LoadTrajectoryError(file_name, "No position data found.")
            if "dt" not in data and "t" not in data:
                raise LoadTrajectoryError(file_name, "No time data found.")

            axes = data.get("axes", None)
            if axes is None:
                logging.warning(
                    "Trajectory '%s' will be loaded but it seems to "
                    "be saved in an old format. Please consider updating it"
                    "by using the JSONSerializer.save method. Older format "
                    "won't be supported in a future.",
                    file_name,
                )
                axes = list(data["r"].values())
            traj_id = data["id"]

            diff_est = data.get("diff_est", None)
            if diff_est is None:
                diff_est = Trajectory.general_diff_est
            else:
                diff_est["method"] = diff.DiffMethod(diff_est["method"])
                diff_est["window_type"] = diff.WindowType(diff_est["window_type"])

            t = data.get("t", None)
            dt = data.get("dt", None)
            t_0 = data.get("t_0", 0.0)

            return Trajectory(
                axes=axes, t=t, dt=dt, t_0=t_0, traj_id=traj_id, diff_est=diff_est
            )
