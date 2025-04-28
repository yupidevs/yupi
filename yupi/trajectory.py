"""
Contains the basic structures for trajectories.
"""

from __future__ import annotations

from typing import (
    Any,
    Collection,
    Iterator,
    NamedTuple,
    Sequence,
    cast,
)

import numpy as np

import yupi._differentiation as diff
from yupi.vector import Vector

_THRESHOLD = 1e-12

Axis = Sequence[float] | np.ndarray
"""Represents the data for a single axis."""

Point = Sequence[float] | np.ndarray
"""Represents a single point."""


class TrajectoryPoint(NamedTuple):
    """
    Represents a point of a trajectory.

    Parameters
    ----------
    r : Vector
        Positional data.
    v : Vector
        Velocity data.
    t : float
        Time data.
    """

    r: Vector
    v: Vector
    t: float


class Trajectory:
    """
    A Trajectory object represents a multidimensional trajectory.
    It can be iterated to obtain the corresponding point for each
    timestep.

    Parameters
    ----------
    x : Axis | None
        Array containing position data of X axis, by default None
    y : Axis | None
        Array containing position data of Y axis, by default None.
    z : Axis | None
        Array containing position data of X axis, by default None.
    points : Sequence[Point] | np.ndarray | None
        Array containing position data as a list of points, by default
        None
    axes : Sequence[Axis] | np.ndarray | None
        Array containing position data as a list of axis, by default
        None
    t : Sequence[float] | np.ndarray | None
        Array containing time data, by default None.
    dt : float | None
        If no time data is given this represents the time between each
        position data value.
    t_0 : float | None
        If no time data is given this represents the initial time value,
        by default 0.
    traj_id : Any
        Id of the trajectory.
    lazy : bool
        Defines if the velocity vector is not recalculated every time
        is asked. By default False.
    diff_est : dict[str, Any]
        Dictionary containing the parameters for the differentiation
        estimation method used to calculate velocity.

    Attributes
    ----------
    r : Vector
        Position vector.
    dt_mean : float
        Mean of the time data delta.
    dt_std : float
        Standard deviation of the time between each position data value.
    traj_id : str
        Id of the trajectory.
    lazy : bool
        Defines if the velocity vector is not recalculated every time
        is asked.
    diff_est : dict
        Dictionary containing the parameters for the differentiation
        estimation method used to calculate velocity.

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

    >>> axes = [[0, 1.2, 3, 2.8], [0, 3.1, 0.7, 1.6]]
    >>> Trajectory(axes=axes)

    All of these examples create the same trajectory.

    Raises
    ------
    ValueError
        If no positional data is given.
    ValueError
        If all the given input data (``x``, ``y``, ``z``, ``t``)
        does not have the same length.
    ValueError
        If ``t`` and ``dt`` given but ``t`` is not uniformly spaced.
    ValueError
        If ``t`` and ``dt`` given but ``dt`` does not match ``t``
        values delta.
    ValueError
        if ``t`` and ``t_0`` are given but ``t_0`` is not the same as
        the first value of ``t``.
    """

    general_diff_est: dict[str, Any] = {
        "method": diff.DiffMethod.LINEAR_DIFF,
        "window_type": diff.WindowType.FORWARD,
    }

    def __init__(
        self,
        x: Axis | None = None,
        y: Axis | None = None,
        z: Axis | None = None,
        points: Sequence[Point] | np.ndarray | None = None,
        axes: Sequence[Axis] | np.ndarray | None = None,
        t: Sequence[float] | np.ndarray | None = None,
        dt: float | None = None,
        t_0: float | None = None,
        traj_id: Any = "",
        lazy: bool = False,
        diff_est: dict[str, Any] | None = None,
    ):
        # Positional data
        self.r: Vector
        self.__init_positional_data(x=x, y=y, z=z, points=points, axes=axes)

        # Time data
        self.__t: Vector | None
        self.__dt: float | None
        self.t_0: float
        self.dt_mean: float
        self.dt_std: float
        self.__init_time_data(t=t, dt=dt, t_0=t_0)

        # Other data
        self.__v: Vector
        self.__a: Vector
        self.traj_id = traj_id
        self.lazy = lazy

        # Differentiation method
        self.diff_est = Trajectory.general_diff_est.copy()
        if diff_est is not None:
            self.diff_est.update(diff_est)

    def __init_positional_data(
        self,
        x: Axis | None = None,
        y: Axis | None = None,
        z: Axis | None = None,
        points: Sequence[Point] | np.ndarray | None = None,
        axes: Sequence[Axis] | np.ndarray | None = None,
    ) -> None:
        """
        Initializes the positional data from the given x, y, and z
        coordinates or from a list of points or axes.

        Parameters
        ----------
        x : Axis | None
            Array containing position data of X axis.
        y : Axis | None
            Array containing position data of Y axis.
        z : Axis | None
            Array containing position data of Z axis.
        points : Sequence[Point] | np.ndarray | None
            Array containing position data as a list of points.
        axes : Sequence[Axis] | np.ndarray | None
            Array containing position data as a list of axes.
        """
        if x is not None:
            self.__init_positional_data_xyz(x, y, z)
        elif points is not None:
            self.__init_positional_data_points(points)
        elif axes is not None:
            self.__init_positional_data_axes(axes)
        else:
            raise ValueError("No positional data were given.")

        if len(self.r) < 2:
            raise ValueError("The trajectory must contain at least 2 points.")

    def __init_positional_data_xyz(
        self,
        x: Axis,
        y: Axis | None = None,
        z: Axis | None = None,
    ) -> None:
        """
        Initializes the positional data from the given x, y, and z
        coordinates.

        Parameters
        ----------
        x : Axis
            Array containing position data of X axis.
        y : Axis | None
            Array containing position data of Y axis.
        z : Axis | None
            Array containing position data of Z axis.
        """
        if y is None and z is not None:
            raise ValueError("If 'x' and 'z' are given, 'y' must be given too.")

        t_len = len(x)
        axis = [data for data in [x, y, z] if data is not None]

        if any(len(data) != t_len for data in axis):
            raise ValueError(
                "All positional data (x, y, and z) must have the same length. "
            )

        self.r = Vector(axis, dtype=float, copy=True).T

    def __init_positional_data_points(
        self,
        points: Sequence[Point] | np.ndarray,
    ) -> None:
        """
        Initializes the positional data from a list of points.

        Parameters
        ----------
        points : Sequence[Point] | np.ndarray
            Array containing position data as a list of points.
        """

        if len(points) < 2:
            raise ValueError("The trajectory must contain at least 2 points.")
        t_dim = len(points[0])
        if any(len(data) != t_dim for data in points):
            raise ValueError(
                "All positional data (points) must have the same length (dimension). "
            )
        self.r = Vector(points, dtype=float, copy=True)

    def __init_positional_data_axes(
        self,
        axes: Sequence[Axis] | np.ndarray,
    ) -> None:
        """
        Initializes the positional data from a list of axes.

        Parameters
        ----------
        axes : list[Axis] | tuple[Axis] | np.ndarray
            List of axes containing the positional data.
        """

        t_len = len(axes[0])
        if any(len(data) != t_len for data in axes):
            raise ValueError("All positional data (axes) must have the same length. ")
        self.r = Vector(axes, dtype=float, copy=True).T

    def __init_time_data(
        self,
        t: Sequence[float] | np.ndarray | None = None,
        dt: float | None = None,
        t_0: float | None = None,
    ) -> None:
        """
        Initializes the time data.

        Parameters
        ----------
        t : Collection[float] | None
            Array containing time data.
        dt : float | None
            If no time data is given this represents the time between
            each position data value.
        t_0 : float | None
            If no time data is given this represents the initial time
            value.
        """

        assert self.r is not None, "Positional data must be initialized first."

        self.__dt = dt
        self.__t = None if t is None else Vector(t, dtype=float, copy=True)

        # Set time data
        if self.__t is None:
            self.dt_mean = dt if dt is not None else 1.0
            self.dt_std = 0
        else:
            if len(self.__t) != len(self.r):
                raise ValueError(
                    "The length of the time data must be the same as "
                    "the length of the position data."
                )
            if t_0 is not None and abs(self.__t[0] - t_0) > _THRESHOLD:
                raise ValueError(
                    "You are giving 't' and 't_0' but 't_0' is not "
                    "the same as the first value of 't'."
                )

            self.dt_mean = np.mean(np.array(self.__t.delta))
            self.dt_std = np.std(np.array(self.__t.delta))

        # Parameters validation
        if self.__t is not None and dt is not None:
            if abs(self.dt_mean - dt) > _THRESHOLD:
                raise ValueError(
                    "You are giving 'dt' and 't' but 'dt' "
                    "does not match with time values delta."
                )
            if abs(self.dt_std - 0) > _THRESHOLD:
                raise ValueError(
                    "You are giving 'dt' and 't' but 't' is not uniformly spaced."
                )

        self.t_0 = t_0 if t_0 is not None else 0.0

    def set_diff_method(
        self,
        method: diff.DiffMethod,
        window_type: diff.WindowType = diff.WindowType.FORWARD,
        accuracy: int = 1,
    ) -> None:
        """
        Set the local diferentiation method.

        Parameters
        ----------
        method : DiffMethod
            Method used to differentiate.
        window_type : WindowType
            Type of window used in the differentiation method. By default,
            the central window is used.
        accuracy : int
            Accuracy of the differentiation method (only valid for
            FORNBERG_DIFF method). By default, the accuracy is 1.
        """
        self.diff_est = {
            "method": method,
            "window_type": window_type,
            "accuracy": accuracy,
        }
        self.recalculate_velocity()

    @staticmethod
    def global_diff_method(
        method: diff.DiffMethod,
        window_type: diff.WindowType = diff.WindowType.FORWARD,
        accuracy: int = 1,
    ) -> None:
        """
        Set the global diferentiation method.

        Parameters
        ----------
        method : DiffMethod
            Method used to differentiate.
        window_type : WindowType
            Type of window used in the differentiation method. By default,
            the central window is used.
        accuracy : int
            Accuracy of the differentiation method (only valid for
            FORNBERG_DIFF method). By default, the accuracy is 1.
        """
        Trajectory.general_diff_est = {
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
            return starts_at_zero and std_is_zero
        return True

    def __len__(self) -> int:
        return self.r.shape[0]

    def __getitem__(self, index: int | slice) -> Trajectory | TrajectoryPoint:
        if isinstance(index, int):
            # r, v, t
            data = [self.r[index], None, None]
            data[1] = self.v[index - 1] if index > 0 else Vector([0] * self.dim)
            data[2] = (
                self.t[index] if self.__t is not None else self.t_0 + index * self.dt
            )

            r, v, t = data
            return TrajectoryPoint(r=r, v=v, t=t)

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            new_points = self.r[start:stop:step]
            if self.uniformly_spaced:
                new_dt = self.dt * step
                new_t0 = self.t_0 + start * self.dt
                return Trajectory(
                    points=new_points,
                    dt=new_dt,
                    t_0=new_t0,
                    diff_est=self.diff_est,
                )
            new_t = self.t[start:stop:step]
            return Trajectory(points=new_points, t=new_t, diff_est=self.diff_est)
        raise TypeError("Index must be an integer or a slice.")

    def __iter__(self) -> Iterator[TrajectoryPoint]:
        for i in range(len(self)):
            yield cast(TrajectoryPoint, self[i])

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """list[tuple[float]] : List of tuples indicanting the min and
        max values of each dimension"""
        _bounds = []
        for dim in range(self.dim):
            min_bound = min(self.r.component(dim))
            max_bound = max(self.r.component(dim))
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

    def recalculate_velocity(self) -> Vector:
        """
        Recalculates the velocity according time data or `dt` if time
        data is not available.

        Returns
        -------
        Vector
            Velocity vector.
        """
        self.__v = diff.estimate_velocity(self, **self.diff_est)
        return self.__v

    def recalculate_acceleration(self) -> Vector:
        """
        Recalculates the acceleration according time data or `dt` if time
        data is not available.

        Returns
        -------
        Vector
            Velocity vector.
        """
        self.__a = diff.estimate_accelereation(self, **self.diff_est)
        return self.__a

    @property
    def v(self) -> Vector:
        """Vector : Velocity vector"""
        if self.lazy and self.__v is not None:
            return self.__v
        return self.recalculate_velocity()

    @property
    def a(self) -> Vector:
        """Vector : Velocity vector"""
        if self.lazy and self.__a is not None:
            return self.__a
        return self.recalculate_acceleration()

    @property
    def t(self) -> Vector:
        """Vector : Time vector"""
        if self.__t is None:
            self.__t = Vector([self.t_0 + self.dt * i for i in range(len(self))])
        return self.__t

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
                "Polar offsets can only be applied on 2 dimensional trajectories"
            )

        # From cartesian to polar
        x, y = self.r.x, self.r.y
        rad, ang = np.hypot(x, y), np.arctan2(y, x)

        rad += radius
        ang += angle

        # From polar to cartesian
        x, y = rad * np.cos(ang), rad * np.sin(ang)
        self.r = Vector([x, y]).T

    def rotate_2d(self, angle: float) -> None:
        """
        Rotates the trajectory around the center coordinates [0,0]

        Parameters
        ----------
        angle : float
            Angle in radians to rotate the trajectory.
        """
        self.add_polar_offset(0, angle)

    def rotate_3d(self, angle: float, vector: Collection[float]) -> None:
        """
        Rotates the trajectory around a given vector.

        Parameters
        ----------
        vector : Collection[float]
            Vector to rotate the trajectory around.
        angle : float
            Angle in radians to rotate the trajectory.

        Raises
        ------
        TypeError
            If the trajectory is not 3 dimensional.
        ValueError
            If the vector has shape different than (3,).
        """
        if self.dim != 3:
            raise TypeError(
                "3D rotations can only be applied on 3 dimensional trajectories"
            )

        vec: Vector = Vector(vector)
        if vec.shape != (3,):
            raise ValueError("The vector must have shape (3,)")

        vec = Vector(vec / vec.norm)
        v_x, v_y, v_z = vec[0], vec[1], vec[2]
        a_cos, a_sin = np.cos(angle), np.sin(angle)

        rot_matrix = np.array(
            [
                [
                    v_x * v_x * (1 - a_cos) + a_cos,
                    v_x * v_y * (1 - a_cos) - v_z * a_sin,
                    v_x * v_z * (1 - a_cos) + v_y * a_sin,
                ],
                [
                    v_x * v_y * (1 - a_cos) + v_z * a_sin,
                    v_y * v_y * (1 - a_cos) + a_cos,
                    v_y * v_z * (1 - a_cos) - v_x * a_sin,
                ],
                [
                    v_x * v_z * (1 - a_cos) - v_y * a_sin,
                    v_y * v_z * (1 - a_cos) + v_x * a_sin,
                    v_z * v_z * (1 - a_cos) + a_cos,
                ],
            ]
        )
        self.r = Vector(np.dot(self.r, rot_matrix))

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
            dt=self.__dt,
            lazy=self.lazy,
            diff_est=self.diff_est,
        )

    def _operable_with(self, other: Trajectory, threshold: float | None = None) -> bool:
        if self.r.shape != other.r.shape:
            return False

        threshold = _THRESHOLD if threshold is None else threshold
        self_time = self.t
        other_time = other.t
        diff = np.abs(np.subtract(self_time, other_time))
        return all(diff < threshold)

    def __iadd__(
        self, other: int | float | tuple | np.ndarray | Trajectory
    ) -> Trajectory:
        if isinstance(other, (int, float)):
            self.r += other
            return self

        if isinstance(other, (list, tuple, np.ndarray)):
            offset = np.array(other, dtype=float)
            if len(offset) != self.dim:
                raise ValueError(
                    "Offset must be the same shape as the other trajectory points"
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

    def __isub__(
        self, other: int | float | tuple | np.ndarray | Trajectory
    ) -> Trajectory:
        if isinstance(other, (int, float)):
            self.r -= other
            return self

        if isinstance(other, (list, tuple, np.ndarray)):
            offset = np.array(other, dtype=float)
            if len(offset) != self.dim:
                raise ValueError(
                    "Offset must be the same shape as the other trajectory points"
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

    def __add__(
        self, other: int | float | tuple | np.ndarray | Trajectory
    ) -> Trajectory:
        traj = self.copy()
        traj += other
        return traj

    def __sub__(
        self, other: int | float | tuple | np.ndarray | Trajectory
    ) -> Trajectory:
        traj = self.copy()
        traj -= other
        return traj

    def __radd__(
        self, other: int | float | tuple | np.ndarray | Trajectory
    ) -> Trajectory:
        return self + other

    def __rsub__(
        self, other: int | float | tuple | np.ndarray | Trajectory
    ) -> Trajectory:
        return self - other

    def __imul__(self, other: int | float) -> Trajectory:
        if isinstance(other, (int, float)):
            self.r *= other
            return self
        raise TypeError(
            "unsoported operation (*) between 'Trajectory' and "
            f"'{type(other).__name__}'"
        )

    def __mul__(self, other: int | float) -> Trajectory:
        traj = self.copy()
        traj *= other
        return traj

    def __rmul__(self, other: int | float) -> Trajectory:
        return self * other

    def turning_angles(
        self,
        accumulate: bool = False,
        degrees: bool = False,
        centered: bool = False,
        wrap: bool = True,
    ) -> np.ndarray:
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
        d_r = self.delta_r
        d_x, d_y = d_r.x, d_r.y
        theta = np.arctan2(d_y, d_x)

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
