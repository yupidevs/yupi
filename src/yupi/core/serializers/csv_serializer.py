"""
CSV traj serializer
"""

from __future__ import annotations

import csv
import logging
import re
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

            # v1 format
            #
            # 0 - trajectory id, dt, dim
            # 1 - method, window_type, accuracy
            # 2 - ...data (point, time)

            # v2 format
            #
            # 0 - version, row where data starts
            # 1 - trajectory id, dt, dim
            # 2 - method, window_type, accuracy
            # 3 - metadata keys
            # 4 - metadata values
            # 5 - extra keys
            # 6 - ...data (point, time, extra1, extra2, ...)

            writer.writerow(["v2", "6"])

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

            if not all((isinstance(v, str) for v in traj.metadata.values())):
                logging.warning(
                    "Not all values in metadata are strings. Converting to"
                    "string for CSV serialization.  It its strongly recommended"
                    "to use other serialization formats that could support more"
                    "complex data types (e.g. JSON)."
                )

            meta_keys = list(traj.metadata.keys())
            meta_vals = [str(traj.metadata[k]) for k in meta_keys]
            writer.writerow(meta_keys)
            writer.writerow(meta_vals)

            extra_keys = list(traj.extra.keys())
            extra_vals = [traj.extra[key] for key in extra_keys]
            writer.writerow(extra_keys)

            # Check extra val types
            if not all((isinstance(v[0], str) for v in extra_vals)):
                logging.warning(
                    "Not all values in extra are strings. Converting to"
                    "string for CSV serialization.  It its strongly recommended"
                    "to use other serialization formats that could support more"
                    "complex data types (e.g. JSON)."
                )

            if extra_vals:
                for p, t, extra in zip(traj.r, traj.t, *extra_vals, strict=True):
                    e_vals = (
                        [str(v) for v in extra]
                        if isinstance(extra, tuple)
                        else [str(extra)]
                    )
                    writer.writerow([*p, t, *e_vals])
            else:
                writer.writerows([*p, t] for p, t in zip(traj.r, traj.t, strict=True))

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

            first_row = next(reader)
            version = 1

            match = re.match(r"^v(\d)", first_row[0])

            if match:
                version = int(match.group(1))

            data_starts_at: int = 3
            traj_id: Any
            dt: float | None
            dim: int | None
            diff_est: dict[str, Any] = {}
            extra_keys: list[str] = []
            extra_vals: list[Any] = []
            metadata: dict[str, Any] = {}

            traj_id, _dt, _dim = first_row if version == 1 else next(reader)

            dt = None if not _dt else float(_dt)
            dim = int(_dim)

            method, window, accuracy = list(map(int, next(reader)))
            diff_est["method"] = diff.DiffMethod(method)
            diff_est["window_type"] = diff.WindowType(window)
            diff_est["accuracy"] = accuracy

            if version == 2:
                data_starts_at = int(first_row[1])
                metadata = dict(zip(next(reader), next(reader), strict=True))
                extra_keys = next(reader)

            assert reader.line_num == data_starts_at, (
                f"Data starts at wrong line {data_starts_at} != {reader.line_num}"
            )
            data = [list(row) for row in reader]

            for i in range(len(extra_keys)):
                extra_vals.append([row[dim + i + 1] for row in data])

            data_arr = np.array([[float(x) for x in row[: dim + 1]] for row in data])
            axes = data_arr[:, :dim].T
            t = data_arr[:, dim]
            return Trajectory(
                axes=axes,
                extra=dict(zip(extra_keys, extra_vals, strict=True)),
                t=t,
                dt=dt,
                traj_id=traj_id,
                diff_est=diff_est,
                **metadata,
            )
