"""
CSV traj serializer
"""

from __future__ import annotations

import csv

import numpy as np

import yupi._differentiation as diff
from yupi import Trajectory
from yupi.core.serializers.serializer import Serializer


class CSVSerializer(Serializer):
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
        CSVSerializer.check_save_path(file_name, overwrite=overwrite, extension=".csv")

        kwargs["encoding"] = kwargs.get("encoding", "utf-8")
        with open(file_name, "w", newline="", **kwargs) as traj_file:
            writer = csv.writer(traj_file, delimiter=",")
            dt = traj.dt if traj.dt_std == 0 else None

            diff_method = Trajectory.general_diff_est.get(
                "method", diff.DiffMethod.LINEAR_DIFF
            )
            diff_win = Trajectory.general_diff_est.get(
                "window_type", diff.WindowType.CENTRAL
            )
            accuracy = Trajectory.general_diff_est.get("accuracy", 1)
            method = traj.diff_est.get("method", diff_method).value
            window = traj.diff_est.get("window_type", diff_win).value
            accuracy = traj.diff_est.get("accuracy", accuracy)

            writer.writerow([traj.traj_id, dt, traj.dim])
            writer.writerow([method, window, accuracy])
            writer.writerows(np.hstack([p, t]) for p, t in zip(traj.r, traj.t))

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

        CSVSerializer.check_load_path(file_name, extension=".csv")

        kwargs["encoding"] = kwargs.get("encoding", "utf-8")
        with open(file_name, "r", **kwargs) as traj_file:
            reader = csv.reader(traj_file, delimiter=",")

            traj_id, dt, dim = next(reader)
            dt = None if not dt else float(dt)
            dim = None if not dim else int(dim)

            method, window, accuracy = list(map(int, next(reader)))
            diff_est = Trajectory.general_diff_est
            diff_est["method"] = diff.DiffMethod(method)
            diff_est["window_type"] = diff.WindowType(window)
            diff_est["accuracy"] = accuracy

            data = np.array([[float(x) for x in row] for row in reader])
            axes = data[:, :dim].T
            t = data[:, dim]
            return Trajectory(axes=axes, t=t, dt=dt, traj_id=traj_id, diff_est=diff_est)
