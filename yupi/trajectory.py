from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import List, NamedTuple, Tuple, Union

import numpy as np

import yupi._vel_estimators as vel_estimators
from yupi.exceptions import LoadTrajectoryError
from yupi.features import Features
from yupi.vector import Vector

_threshold = 1e-12


class TrajectoryPoint(NamedTuple):
    """
    Represents a point of a trajectory.

    Parameters
    ----------
    r : Vector
        Positional data.
    ang : Vector
        Angular data.
    v : Vector
        Velocity data.
    t : float
        Time data.
    """

    r: Vector
    ang: Vector
    v: Vector
    t: float


class Trajectory:
    """
    A Trajectory object represents a multidimensional trajectory.
    It can be iterated to obtain the corresponding point for each
    timestep.

    Parameters
    ----------
    x : np.ndarray
        Array containing position data of X axis, by default None
    y : np.ndarray
        Array containing position data of Y axis, by default None.
    z : np.ndarray
        Array containing position data of X axis, by default None.
    points : np.ndarray
        Array containing position data as a list of points, by default
        None
    dimensions : np.ndarray
        Array containing position data as a list of axis, by default
        None
    t : np.ndarray
        Array containing time data, by default None.
    ang : np.ndarray
        Array containing angle data, by default None.
    dt : float
        If no time data is given this represents the time between each
        position data value.
    t0 : float
        If no time data is given this represents the initial time value,
        by default 0.
    traj_id : str
        Id of the trajectory.
    lazy : bool
        Defines if the velocity vector is not recalculated every time
        is asked.
    vel_est : dict
        Dictionary containing the parameters for the velocity estimation
        method.

    Attributes
    ----------
    dt : float
        If no time data is given this represents the time between each
        position data value.
    id : str
        Id of the trajectory.

    Examples
    --------
    You can create a trajectory object by giving the arrays that
    represent it:

    >>> x = [0, 1.2, 3, 2.8]
    >>> y = [0, 3.1, 0.7, 1.6]
    >>> Trajectory(x=x, y=y)

    You can also create the trajectory given the points:

    >>> points = [[0, 0], [1.2, 3.1], [3, 0.7], [2.8, 1.6]]
    >>> Trajectory(points=points)

    Or even create it given all the data for each dimension in a single
    source:

    >>> dims = [[0, 1.2, 3, 2.8], [0, 3.1, 0.7, 1.6]]
    >>> Trajectory(dimensions=dims)

    All of these examples create the same trajectory.

    Raises
    ------
    ValueError
        If positional data is given in more than one way.
    ValueError
        If no positional data is given.
    ValueError
        If all the given input data (``x``, ``y``, ``z``, ``t``,
        ``ang``) does not have the same shape.
    ValueError
        If ``t`` and ``dt`` given but ``t`` is not uniformly spaced.
    ValueError
        If ``t`` and ``dt`` given but ``dt`` does not match ``t``
        values delta.
    """

    __vel_est = {
        "method": vel_estimators.VelocityMethod.LINEAR_DIFF,
        "window_type": vel_estimators.WindowType.CENTRAL,
    }

    def __init__(
        self,
        x: np.ndarray = None,
        y: np.ndarray = None,
        z: np.ndarray = None,
        points: np.ndarray = None,
        dimensions: np.ndarray = None,
        t: np.ndarray = None,
        ang: np.ndarray = None,
        dt: float = None,
        t0: float = 0.0,
        traj_id: str = None,
        lazy: bool = False,
        vel_est: dict = None,
    ):

        # Position data validation
        from_xyz = x is not None
        from_points = points is not None
        from_dimensions = dimensions is not None

        if from_xyz + from_points + from_dimensions > 1:
            raise ValueError(
                "Positional data must come only from one way: "
                "'xyz' data, 'points' data or 'dimensions' data."
            )

        # Set position data
        self.r = None
        data: List[Vector] = [t, ang]
        lengths = [len(item) for item in data if item is not None]

        for i, item in enumerate(data):
            if item is not None:
                data[i] = Vector.create(item, dtype=float)

        if from_xyz:
            dimensions = [d for d in [x, y, z] if d is not None]
            from_dimensions = True

        if from_dimensions and len(dimensions) > 0:
            lengths.extend([len(d) for d in dimensions])
            self.r = Vector.create(dimensions, dtype=float).T

        if from_points:
            lengths.append(len(points))
            self.r = Vector.create(points, dtype=float)

        if self.r is None:
            raise ValueError("No position data were given.")

        if lengths.count(lengths[0]) != len(lengths):
            raise ValueError("All input arrays must have the same shape.")

        self.__dt = dt
        self.dt_mean = dt
        self.__t0 = t0
        self.__t = data[0]
        self.ang = data[1]
        self.traj_id = traj_id
        self.lazy = lazy

        # Set time data
        if self.__t is None:
            self.dt_mean = dt if dt is not None else 1.0
            self.dt_std = 0
        else:
            self.dt_mean = np.mean(np.array(self.__t.delta))
            self.dt_std = np.std(np.array(self.__t.delta))

        # Velocity estimation
        self.vel_est = Trajectory.__vel_est.copy()
        if vel_est is not None:
            self.vel_est.update(vel_est)

        self.recalculate_velocity()

        # Time parameters validation
        if t is not None and dt is not None:
            if abs(self.dt_mean - dt) > _threshold:
                raise ValueError(
                    "You are giving 'dt' and 't' but 'dt' "
                    "does not match with time values delta."
                )
            if abs(self.dt_std - 0) > _threshold:
                raise ValueError(
                    "You are giving 'dt' and 't' but 't' is " "not uniformly spaced."
                )
            if abs(self.__t[0] - t0) > _threshold:
                raise ValueError(
                    "You are giving 'dt' and 't' but 't0' is not "
                    "the same as the first value of 't'."
                )

        self.features = Features(self)

    def set_vel_method(
        self,
        method: vel_estimators.VelocityMethod,
        window_type: vel_estimators.WindowType = vel_estimators.WindowType.CENTRAL,
        accuracy: int = 1,
    ):
        """
        Set the method to calculate the velocity.

        Parameters
        ----------
        method : VelocityMethod
            Method to calculate the velocity.
        window_type : WindowType
            Type of window to use to calculate the velocity. By default,
            the central window is used.
        accuracy : int
            Accuracy of the velocity estimation (only valid for
            FORNBERG_DIFF method). By default, the accuracy is 1.
        """
        self.vel_est = {
            "method": method,
            "window_type": window_type,
            "accuracy": accuracy,
        }
        self.recalculate_velocity()

    @staticmethod
    def global_vel_method(
        method: vel_estimators.VelocityMethod,
        window_type: vel_estimators.WindowType = vel_estimators.WindowType.CENTRAL,
        accuracy: int = 1,
    ):
        """
        Set the method to calculate the velocity.

        Parameters
        ----------
        method : VelocityMethod
            Method to calculate the velocity.
        window_type : WindowType
            Type of window to use to calculate the velocity. By default,
            the central window is used.
        accuracy : int
            Accuracy of the velocity estimation (only valid for
            FORNBERG method). By default, the accuracy is 1.
        """
        Trajectory.__vel_est = {
            "method": method,
            "window_type": window_type,
            "accuracy": accuracy,
        }

    @property
    def dt(self) -> float:
        """
        Returns the time between each position data value.

        If the time data is not uniformly spaced it returns an
        estimated value.
        """
        return self.dt_mean if self.__dt is None else self.__dt

    @property
    def uniformly_spaced(self) -> bool:
        """bool : True if the time data is uniformly spaced"""
        if self.__t is not None:
            starts_at_zero = self.__t[0] == 0
            std_is_zero = self.dt_std == 0
            if starts_at_zero and std_is_zero:
                return True
            return False
        return True

    def __len__(self):
        return self.r.shape[0]

    def __getitem__(self, index):
        if isinstance(index, int):
            # *dim, *ang, v, t
            data = [self.r[index], [], None, None]

            # Angle
            if self.ang is not None:
                data[1] = self.ang[index]

            # Velocity
            data[2] = self.v[index - 1] if index > 0 else Vector.create([0] * self.dim)

            # Time
            if self.__t is not None:
                data[3] = self.__t[index]
            else:
                data[3] = self.__t0 + self.dt * index

            r, ang, v, t = data
            return TrajectoryPoint(r=r, ang=ang, v=v, t=t)

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))

            new_points = self.r[start:stop:step]
            new_ang = None
            if self.ang is not None:
                new_ang = self.ang[start:stop:step]
            if self.uniformly_spaced:
                new_dt = self.dt * step
                new_t0 = self.__t0 + start * self.dt
                return Trajectory(
                    points=new_points,
                    ang=new_ang,
                    dt=new_dt,
                    t0=new_t0,
                    vel_est=self.vel_est,
                )
            new_t = self.t[start:stop:step]
            return Trajectory(
                points=new_points, ang=new_ang, t=new_t, vel_est=self.vel_est
            )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def bounds(self) -> List[Tuple[float]]:
        """List[Tuple[float]] : List of tuples indicanting the min and
        max values of each dimension"""
        _bounds = []
        for d in range(self.dim):
            min_bound = min(self.r.component(d))
            max_bound = max(self.r.component(d))
            _bounds.append((min_bound, max_bound))
        return _bounds

    @property
    def dim(self) -> int:
        """int : Trajectory spacial dimensions."""
        return self.r.shape[1]

    @property
    def delta_r(self) -> Vector:
        """Vector: Difference between each couple of consecutive points
        in the Trajectory."""
        return self.r.delta

    @property
    def delta_v(self) -> Vector:
        """Vector: Difference between each couple of consecutive sample
        in the velocity vector of the Trajectory."""
        return self.v.delta

    @property
    def delta_ang(self) -> Union[Vector, None]:
        """Union[Vector, None] : Difference between each couple of
        consecutive samples in the ``ang`` vector of the Trajectory."""
        if self.ang is not None:
            return self.ang.delta
        return None

    def recalculate_velocity(self) -> Vector:
        """
        Recalculates the velocity according time data or `dt` if time
        data is not available.

        Returns
        -------
        Vector
            Velocity vector.
        """

        self.__v = vel_estimators.estimate_velocity(self, **self.vel_est)
        return self.__v

    @property
    def v(self) -> Vector:
        """Vector : Velocity vector"""
        if self.__v is None:
            method = self.vel_est["method"]
            win_type = self.vel_est.get(
                "window_type", vel_estimators.WindowType.CENTRAL
            )
            acc = self.vel_est.get("accuracy", 1)
            acc_text = (
                f" and accuracy {acc}"
                if method == vel_estimators.VelocityMethod.FORNBERG_DIFF
                else ""
            )
            raise ValueError(
                f"Trajectory velocity can not be estimated using {method.name} "
                f"method with window type {win_type.name}{acc_text}."
            )
        if self.lazy:
            return self.__v
        return self.recalculate_velocity()

    @property
    def t(self) -> Vector:
        """Vector : Time vector"""
        if self.__t is None:
            dt_vec = [self.__t0 + self.dt * i for i in range(len(self))]
            self.__t = Vector.create(dt_vec)
        return self.__t

    @property
    def ang_velocity(self) -> Union[Vector, None]:
        """Union[Vector, None] : Computes the angular velocity from the
        ``ang`` vector of the Trajectory."""
        if self.ang is not None:
            return self.ang.delta / self.dt
        return None

    def add_polar_offset(self, radius: float, angle: float) -> None:
        """
        Adds an offset given a point in polar coordinates.

        Parameters
        ----------
        radius : float
            Point's radius.
        angle : float
            Point's angle.

        Raises
        ------
        TypeError
            If the trajectory is not 2 dimensional.
        """

        if self.dim != 2:
            raise TypeError(
                "Polar offsets can only be applied on 2 " "dimensional trajectories"
            )

        # From cartesian to polar
        x, y = self.r.x, self.r.y
        rad = np.hypot(x, y)
        ang = np.arctan2(y, x)

        rad += radius
        ang += angle

        # From polar to cartesian
        x = rad * np.cos(ang)
        y = rad * np.sin(ang)
        self.r = Vector.create([x, y]).T

    def rotate2d(self, angle: float):
        """
        Rotates the trajectory around the center coordinates [0,0]

        Parameters
        ----------
        angle : float
            Angle in radians to rotate the trajectory.
        """
        self.add_polar_offset(0, angle)

    def rotate3d(self, angle: float, vector: Union[list, np.ndarray]):
        """
        Rotates the trajectory around a given vector.

        Parameters
        ----------
        vector : Vector
            Vector to rotate the trajectory around.
        angle : float
            Angle in radians to rotate the trajectory.

        Raises
        ------
        TypeError
            If the trajectory is not 3 dimensional.
        """

        if self.dim != 3:
            raise TypeError(
                "3D rotations can only be applied on 3 " "dimensional trajectories"
            )

        vec: Vector = Vector.create(vector)
        if len(vec) != 3:
            raise ValueError("The vector must have 3 components")

        vec = vec / vec.norm
        vx, vy, vz = vec[0], vec[1], vec[2]
        c, s = np.cos(angle), np.sin(angle)

        R = np.array(
            [
                [
                    vx * vx * (1 - c) + c,
                    vx * vy * (1 - c) - vz * s,
                    vx * vz * (1 - c) + vy * s,
                ],
                [
                    vx * vy * (1 - c) + vz * s,
                    vy * vy * (1 - c) + c,
                    vy * vz * (1 - c) - vx * s,
                ],
                [
                    vx * vz * (1 - c) - vy * s,
                    vy * vz * (1 - c) + vx * s,
                    vz * vz * (1 - c) + c,
                ],
            ]
        )

        self.r = Vector.create(np.dot(self.r, R))

    def copy(self) -> Trajectory:
        """
        Returns a copy of the trajectory.

        Returns
        -------
        Trajectory
            Copy of the trajectory.
        """

        return Trajectory(
            points=self.r,
            t=self.__t,
            ang=self.ang,
            dt=self.__dt,
            lazy=self.lazy,
            vel_est=self.vel_est,
        )

    def _operable_with(self, other: Trajectory, threshold=None) -> bool:
        if threshold is None:
            threshold = _threshold

        if self.r.shape != other.r.shape:
            return False

        self_time = self.t
        other_time = other.t

        diff = np.abs(self_time - other_time)
        return all(diff < threshold)

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            self.r += other
            return self

        if isinstance(other, (list, tuple, np.ndarray)):
            offset = np.array(other, dtype=float)
            if len(offset) != self.dim:
                raise ValueError(
                    "Offset must be the same shape as the other " "trajectory points"
                )
            self.r += offset
            return self

        if isinstance(other, Trajectory):
            if not self._operable_with(other):
                raise ValueError("Incompatible trajectories")
            self.r += other.r
            return self

        raise TypeError(
            "unsoported operation (+) between 'Trajectory' and "
            f"'{type(other).__name__}'"
        )

    def __isub__(self, other):
        if isinstance(other, (int, float)):
            self.r -= other
            return self

        if isinstance(other, (list, tuple, np.ndarray)):
            offset = np.array(other, dtype=float)
            if len(offset) != self.dim:
                raise ValueError(
                    "Offset must be the same shape as the other " "trajectory points"
                )
            self.r -= offset
            return self

        if isinstance(other, Trajectory):
            if not self._operable_with(other):
                raise ValueError("Incompatible trajectories")
            self.r -= other.r
            return self

        raise TypeError(
            "unsoported operation (-) between 'Trajectory' and "
            f"'{type(other).__name__}'"
        )

    def __add__(self, other):
        traj = self.copy()
        traj += other
        return traj

    def __sub__(self, other):
        traj = self.copy()
        traj -= other
        return traj

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self - other

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self.r *= other
            return self

        raise TypeError(
            "unsoported operation (*) between 'Trajectory' and "
            f"'{type(other).__name__}'"
        )

    def __mul__(self, other):
        traj = self.copy()
        traj *= other
        return traj

    def __rmul__(self, other):
        return self * other

    def turning_angles(
        self, accumulate=False, degrees=False, centered=False, wrap=True
    ):
        """
        Return the sequence of turning angles that forms the trajectory.

        Parameters
        ----------
        traj : Trajectory
            Input trajectory.
        accumulate : bool, optional
            If True, turning angles are measured with respect to an axis
            defined by the initial velocity (i.e., angles between initial
            and current velocity). Otherwise, relative turning angles
            are computed (i.e., angles between succesive velocity
            vectors). By default False.
        degrees : bool, optional
            If True, angles are given in degrees. Otherwise, the units
            are radians. By default False.
        centered : bool, optional
            If True, angles are wrapped on the interval ``[-pi, pi]``.
            Otherwise, the interval ``[0, 2*pi]`` is chosen. By default
            False.
        wrap : bool, optional
            If True, angles are wrapped in a certain interval (depending
            on ``centered`` param). By default True.

        Returns
        -------
        np.ndarray
            Turning angles where each position in the array correspond
            to a given time instant.
        """

        dx = self.delta_r.x
        dy = self.delta_r.y
        theta = np.arctan2(dy, dx)

        if not accumulate:
            theta = np.ediff1d(theta)  # Relative turning angles
        else:
            theta -= theta[0]  # Accumulative turning angles

        if degrees:
            theta = np.rad2deg(theta)

        if not wrap:
            return theta

        discont = 360 if degrees else 2 * np.pi
        if not centered:
            return theta % discont

        discont_half = discont / 2
        return -((discont_half - theta) % discont - discont_half)

    def _save_json(self, path: str):
        def convert_to_list(vec: Vector):
            if vec is None:
                return vec

            if len(vec.shape) == 1:
                return list(vec)
            return {d: list(v) for d, v in enumerate(vec)}

        ang = None if self.ang is None else self.ang.T

        default_vel_est_method = vel_estimators.VelocityMethod.LINEAR_DIFF
        default_vel_est_window = vel_estimators.WindowType.CENTRAL
        default_vel_est_accuracy = 1
        vel_est = {
            "method": self.vel_est.get("method", default_vel_est_method).value,
            "window_type": self.vel_est.get("window", default_vel_est_window).value,
            "accuracy": self.vel_est.get("accuracy", default_vel_est_accuracy),
        }

        json_dict = {
            "id": self.traj_id,
            "dt": self.__dt,
            "r": convert_to_list(self.r.T),
            "ang": convert_to_list(ang),
            "t": convert_to_list(self.__t),
            "vel_est": vel_est,
        }
        with open(str(path), "w") as traj_file:
            json.dump(json_dict, traj_file)

    def _save_csv(self, path):
        with open(str(path), "w", newline="") as traj_file:
            writer = csv.writer(traj_file, delimiter=",")
            ang_shape = 0 if self.ang is None else self.ang.shape[1]
            writer.writerow([self.traj_id, self.__dt, self.dim, ang_shape])

            default_vel_est_method = vel_estimators.VelocityMethod.LINEAR_DIFF
            default_vel_est_window = vel_estimators.WindowType.CENTRAL
            default_vel_est_accuracy = 1
            method = self.vel_est.get("method", default_vel_est_method).value
            window = self.vel_est.get("window", default_vel_est_window).value
            accuracy = self.vel_est.get("accuracy", default_vel_est_accuracy)
            print(method, window, accuracy)
            writer.writerow([method, window, accuracy])

            for tp in self:
                row = np.hstack(np.array([tp.r, tp.ang, tp.t], dtype=object))
                writer.writerow(row)

    def save(
        self,
        file_name: str,
        path: str = ".",
        file_type: str = "json",
        overwrite: bool = True,
    ):
        """
        Saves the trajectory to disk.

        Parameters
        ----------
        file_name : str
            Name of the file.
        path : str
            Path where to save the trajectory, by default ``'.'``.
        file_time : str
            Type of the file, by default ``json``.

            The only types avaliable are: ``json`` and ``csv``.
        overwrite : bool
            Wheter or not to overwrite the file if it already exists,
            by default True.

        Raises
        ------
        ValueError
            If ``override`` parameter is ``False`` and the file already
            exists.
        ValueError
            If ``file_type`` is not ``json`` or ``csv``.

        Examples
        --------
        >>> t = Trajectory(x=[0.37, 1.24, 1.5])
        >>> t.save('my_track')
        """

        # Build full path
        full_path = Path(path) / Path(f"{file_name}.{file_type}")

        # Check file existance
        if not overwrite and full_path.exists():
            raise ValueError(f"File '{str(full_path)}' already exist")

        if file_type == "json":
            self._save_json(full_path)
        elif file_type == "csv":
            self._save_csv(full_path)
        else:
            raise ValueError(f"Invalid export file type '{file_type}'")

    @staticmethod
    def save_trajectories(
        trajectories: list,
        folder_path: str = ".",
        file_type: str = "json",
        overwrite: bool = True,
    ):
        """
        Saves a list of trajectories to disk. Each Trajectory object
        will be saved in a separate file inside the given folder.

        Parameters
        ----------
        trajectories : list[Trajectory]
            List of Trajectory objects that will be saved.
        folder_path : str
            Path where to save all the trajectory, by default ``'.'``.
        file_type : str
            Type of the file, by default ``json``.

            The only types avaliable are: ``json`` and ``csv``.
        overwrite : bool
            Wheter or not to overwrite the file if it already exists,
            by default True.

        Examples
        --------
        >>> t1 = Trajectory(x=[0.37, 1.24, 1.5])
        >>> t2 = Trajectory(x=[1, 2], y=[3, 4])
        >>> Trajectory.save_trajectories([t1, t2])
        """

        for i, traj in enumerate(trajectories):
            path = str(Path(folder_path))
            name = str(Path(f"trajectory_{i}"))
            traj.save(name, path, file_type, overwrite)

    @staticmethod
    def _load_json(path: str):
        with open(path, "r") as traj_file:
            data = json.load(traj_file)

            traj_id = data["id"]
            dt = data["dt"]
            t = data["t"]
            ang = None

            if data["ang"] is not None:
                ang_values = list(data["ang"].values())
                ang = Vector.create(ang_values).T

            dims = list(data["r"].values())
            vel_est = data.get("vel_est", None)
            if vel_est is None:
                vel_est = Trajectory.__vel_est
            else:
                vel_est["method"] = vel_estimators.VelocityMethod(vel_est["method"])
                vel_est["window_type"] = vel_estimators.WindowType(
                    vel_est["window_type"]
                )

            return Trajectory(
                dimensions=dims, t=t, ang=ang, dt=dt, traj_id=traj_id, vel_est=vel_est
            )

    @staticmethod
    def _load_csv(path: str):
        with open(path, "r") as traj_file:

            def check_empty_val(val, cast=True):
                if val == "":
                    return None
                return float(val) if cast else val

            r, ang, t = [], [], []
            traj_id, dt, dim = None, None, None

            for i, row in enumerate(csv.reader(traj_file)):
                if i == 0:
                    traj_id = check_empty_val(row[0], cast=False)
                    dt = check_empty_val(row[1])
                    dim = int(row[2])
                    ang_dim = int(row[3])
                    r = [[] for _ in range(dim)]
                    ang = [[] for _ in range(ang_dim)]
                    continue

                if i == 1:
                    vel_est = {
                        "method": vel_estimators.VelocityMethod(int(row[0])),
                        "window_type": vel_estimators.WindowType(int(row[1])),
                        "accuracy": int(row[2]),
                    }
                    continue

                for j in range(dim):
                    r[j].append(float(row[j]))

                for j, k in enumerate(range(dim, dim + ang_dim)):
                    ang[j] = row[k]

                t.append(float(row[-1]))

            if not ang:
                ang = None

            return Trajectory(dimensions=r, t=t, ang=ang, dt=dt, traj_id=traj_id)

    @staticmethod
    def load(file_path: str):
        """
        Loads a trajectory

        Parameters
        ----------
        file_path : str
            Path of the trajectory file

        Returns
        -------
        Trajectory
            Loaded Trajectory object.

        Raises
        ------
        ValueError
            If ``file_path`` is a non existing path.
        ValueError
            If ``file_path`` is a not a file.
        ValueError
            If ``file_path`` extension is not ``json`` or ```csv``.
        """

        path = Path(file_path)

        # Check valid path
        if not path.exists():
            raise ValueError("Path does not exist.")
        if not path.is_file():
            raise ValueError("Path must be a file.")

        file_type = path.suffix

        try:
            if file_type == ".json":
                return Trajectory._load_json(file_path)
            elif file_type == ".csv":
                return Trajectory._load_csv(file_path)
            else:
                raise ValueError("Invalid file type.")
        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as exc:
            raise LoadTrajectoryError(path) from exc

    @staticmethod
    def load_folder(folder_path=".", recursively: bool = False):
        """
        Loads all the trajectories from a folder.

        Parameters
        ----------
        folder_path : str
            Path of the trajectories folder.
        recursively : bool
            If True then subfolders are analized recursively, by
            default False.

        Returns
        -------
        List[Trajectory]
            List of the loaded trajectories.
        """

        trajectories = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                path = str(Path(root) / Path(file))
                try:
                    trajectories.append(Trajectory.load(path))
                except LoadTrajectoryError as load_exception:
                    print(f"Ignoring: {load_exception.path}")
            if not recursively:
                break
        return trajectories
