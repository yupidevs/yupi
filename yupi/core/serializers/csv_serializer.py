"""
CSV traj serializer
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

import yupi._differentiation as diff
from yupi.core.serializers.serializer import Serializer
from yupi.trajectory import Trajectory


class CSVSerializer(Serializer):
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

        CSVSerializer.check_save_path(_path, overwrite=overwrite, extension=".csv")

        kwargs["encoding"] = kwargs.get("encoding", "utf-8")
        with _path.open("w", newline="", **kwargs) as traj_file:
            writer = csv.writer(traj_file, delimiter=",")
            dt = traj.dt if traj.dt_std == 0 else None

            diff_method = Trajectory.general_diff_est.get(
                "method", diff.DiffMethod.LINEAR_DIFF
            )
            diff_win = Trajectory.general_diff_est.get(
                "window_type", diff.WindowType.FORWARD
            )
            accuracy = Trajectory.general_diff_est.get("accuracy", 1)
            method = traj.diff_est.get("method", diff_method).value
            window = traj.diff_est.get("window_type", diff_win).value
            accuracy = traj.diff_est.get("accuracy", accuracy)

            writer.writerow([traj.traj_id, dt, traj.dim])
            writer.writerow([method, window, accuracy])
            writer.writerows(
                np.hstack([p, t]) for p, t in zip(traj.r, traj.t, strict=True)
            )

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

        CSVSerializer.check_load_path(_path, extension=".csv")

        kwargs["encoding"] = kwargs.get("encoding", "utf-8")
        with _path.open("r", **kwargs) as traj_file:
            reader = csv.reader(traj_file, delimiter=",")

            traj_id, _dt, _dim = next(reader)
            dt = None if not _dt else float(_dt)
            dim = None if not _dim else int(_dim)

            method, window, accuracy = list(map(int, next(reader)))
            diff_est = Trajectory.general_diff_est
            diff_est["method"] = diff.DiffMethod(method)
            diff_est["window_type"] = diff.WindowType(window)
            diff_est["accuracy"] = accuracy

            data = np.array([[float(x) for x in row] for row in reader])
            axes = data[:, :dim].T
            t = data[:, dim]
            return Trajectory(axes=axes, t=t, dt=dt, traj_id=traj_id, diff_est=diff_est)
