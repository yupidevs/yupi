from __future__ import annotations
import json
import csv
import os
from pathlib import Path
from typing import List, NamedTuple, Union
import numpy as np

from yupi.vector import Vector
from yupi.exceptions import LoadTrajectoryError


_threshold = 1e-12


class TrajectoryPoint(NamedTuple):
    """Represents a point of a trajectory."""

    r: Vector
    ang: Vector
    v: Vector
    t: float


class Trajectory():
    """
    A Trajectory object represents a multidimensional trajectory.
    It can be iterated to obtain the corresponding point for each
    timestep.

    Parameters
    ----------
    x : np.ndarray
        Array containing position data of X axis.
    y : np.ndarray
        Array containing position data of Y axis, by default None.
    z : np.ndarray
        Array containing position data of X axis, by default None.
    points : np.ndarray
        Array containing position data as a list of points.
    dimensions : np.ndarray
        Array containing position data as a list of axis.
    t : np.ndarray
        Array containing time data, by default None.
    ang : np.ndarray
        Array containing angle data, by default None.
    dt : float
        If no time data is given this represents the time between each
        position data value.
    traj_id : str
        Id of the trajectory.
    lazy : bool
        Defines if the velocity vector is not recalculated every time
        is asked.

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
        If ``x`` is not given.
    ValueError
        If all the given input data (``x``, ``y``, ``z``, ``t``, ``ang``)
        does not have the same shape.
    """

    def __init__(self, x: np.ndarray = None, y: np.ndarray = None,
                 z: np.ndarray = None, points: np.ndarray = None,
                 dimensions: np.ndarray = None, t: np.ndarray = None,
                 ang: np.ndarray = None, dt: float = 1.0,
                 traj_id: str = None, lazy: bool = False):

        from_xyz = x is not None
        from_points = points is not None
        from_dimensions = dimensions is not None

        if from_xyz + from_points + from_dimensions > 1:
            raise ValueError("Positional data must come only from one way: "
                             "'xyz' data, 'points' data or 'dimensions' data.")

        self.r = None
        data: List[Vector] = [t, ang]
        lengths = [len(item) for item in data if item is not None]

        for i, item in enumerate(data):
            if item is not None:
                data[i] = Vector.create(item, dtype=float)

        if from_xyz:
            dimensions = [d for d in [x, y, z] if d is not None]
            from_dimensions = True

        if from_dimensions:
            if len(dimensions) == 0:
                raise ValueError('Trajectory requires at least one dimension.')
            lengths.extend([len(d) for d in dimensions])
            self.r = Vector.create(dimensions, dtype=float).T

        if from_points:
            lengths.append(len(points))
            self.r = Vector.create(points, dtype=float)

        if lengths.count(lengths[0]) != len(lengths):
            raise ValueError('All input arrays must have the same shape.')

        if self.r is None:
            raise ValueError('No position data were given.')

        self.t = data[0]
        self.ang = data[1]
        self.id = traj_id
        self.lazy = lazy

        if self.t is None:
            self.dt = dt
            self.dt_std = 0
            self.__v: Vector = self.r.delta / self.dt
        else:
            self.dt = np.mean(np.array(self.t.delta))
            self.dt_std = np.std(np.array(self.t.delta))
            self.__v: Vector = (self.r.delta.T / self.t.delta).T

    def __len__(self):
        return self.r.shape[0]

    def __iter__(self):
        current_time = 0
        for i in range(len(self)):
            # *dim, *ang, v, t
            data = [self.r[i], [], None, None]

            # Angle
            if self.ang is not None:
                data[1] = self.ang[i]

            # Velocity
            data[2] = self.v[i - 1] if i > 0 else Vector.create([0]*self.dim)

            # Time
            if self.t is not None:
                data[3] = self.t[i]
            else:
                data[3] = current_time
                current_time += self.dt

            r, ang, v, t = data
            yield TrajectoryPoint(r=r, ang=ang, v=v, t=t)

    @property
    def dim(self) -> int:
        """int : Trajectory spacial dimensions."""
        return self.r.shape[1]

    @property
    def delta_t(self) -> Vector:
        """Vector : Difference between each couple of consecutive
        samples in the Trajectory."""
        if self.t is not None:
            return self.t.delta
        return None

    @property
    def delta_r(self) -> Vector:
        """Vector: Difference between each couple of consecutive points
        in the Trajectory."""
        return self.r.delta

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

        if self.t is None:
            self.__v: Vector = self.r.delta / self.dt
        else:
            self.__v: Vector = (self.r.delta.T / self.t.delta).T
        return self.__v

    @property
    def v(self) -> Vector:
        """Vector : Velocity vector"""
        if self.lazy:
            return self.__v
        return self.recalculate_velocity()

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
            raise TypeError('Polar offsets can only be applied on 2 '
                            'dimensional trajectories')

        # From cartesian to polar
        x, y = self.r.x, self.r.y
        rad = np.hypot(x, y)
        ang = np.arctan2(y, x)

        rad += radius
        ang += angle

        # From polar to cartesian
        x = rad * np.cos(ang)
        y = rad * np.sin(ang)
        self.r = Vector.create([x,y]).T

    def copy(self) -> Trajectory:
        """
        Returns a copy of the trajectory.

        Returns
        -------
        Trajectory
            Copy of the trajectory.
        """

        return Trajectory(points=self.r, t=self.t, ang=self.ang, dt=self.dt,
                          lazy=self.lazy)

    def _operable_with(self, other: Trajectory, threshold=None) -> bool:
        if threshold is None:
            threshold = _threshold

        if self.r.shape != other.r.shape:
            return False

        self_time = self.t
        if self_time is None:
            self_time = np.array([self.dt*i for i in range(len(self))])

        other_time = other.t
        if other_time is None:
            other_time = np.array([other.dt*i for i in range(len(self))])

        diff = np.abs(self_time - other_time)
        return all(diff < threshold)

    def __iadd__(self, other):
        if isinstance(other, (list, tuple, np.ndarray)):
            offset = np.array(other)
            if len(offset) != self.dim:
                raise ValueError('Offset must be the same shape as the other '
                                 'trajectory points')
            self.r += offset
            return self

        if isinstance(other, Trajectory):
            if not self._operable_with(other):
                raise ValueError('Incompatible trajectories')
            self.r += other.r
            return self

        raise TypeError("unsoported operation (+) between 'Trajectory' and "
                        f"'{type(other).__name__}'")

    def __isub__(self, other):
        if isinstance(other, (list, tuple, np.ndarray)):
            offset = np.array(other, dtype=float)
            return self + (-1 * offset)
        if isinstance(other, Trajectory):
            if not self._operable_with(other):
                raise ValueError('Incompatible trajectories')
            self.r -= other.r
            return self
        raise TypeError("unsoported operation (-) between 'Trajectory' and "
                        f"'{type(other).__name__}'")

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

    def _save_json(self, path: str):
        def convert_to_list(vec: Vector):
            if vec is None:
                return vec

            if len(vec.shape) == 1:
                return list(vec)
            return {d: list(v) for d, v in enumerate(vec)}

        ang = None if self.ang is None else self.ang.T
        json_dict = {
            'id': self.id,
            'dt': self.dt,
            'r': convert_to_list(self.r.T),
            'ang': convert_to_list(ang),
            't': convert_to_list(self.t)
        }
        with open(str(path), 'w') as traj_file:
            json.dump(json_dict, traj_file)

    def _save_csv(self, path):
        with open(str(path), 'w', newline='') as traj_file:
            writer = csv.writer(traj_file, delimiter=',')
            ang_shape = 0 if self.ang is None else self.ang.shape[1]
            writer.writerow([self.id, self.dt, self.dim, ang_shape])
            for tp in self:
                row = np.hstack(np.array([tp.r, tp.ang, tp.t]))
                writer.writerow(row)

    def save(self, file_name: str, path: str = '.', file_type: str = 'json',
             overwrite: bool = True):
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
        full_path = Path(path) / Path(f'{file_name}.{file_type}')

        # Check file existance
        if not overwrite and full_path.exists():
            raise ValueError(f"File '{str(full_path)}' already exist")

        if file_type == 'json':
            self._save_json(full_path)
        elif file_type == 'csv':
            self._save_csv(full_path)
        else:
            raise ValueError(f"Invalid export file type '{file_type}'")

    @staticmethod
    def save_trajectories(trajectories: list, folder_path: str = '.',
                          file_type: str = 'json', overwrite: bool = True):
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
            name = str(Path(f'trajectory_{i}'))
            traj.save(name, path, file_type, overwrite)

    @staticmethod
    def _load_json(path: str):
        with open(path, 'r') as traj_file:
            data = json.load(traj_file)

            traj_id = data['id']
            dt = data['dt']
            t = data['t']
            ang = None

            if data['ang'] is not None:
                ang_values = list(data['ang'].values())
                ang = Vector.create(ang_values).T

            dims = list(data['r'].values())

            return Trajectory(dimensions=dims, t=t, ang=ang, dt=dt,
                              traj_id=traj_id)

    @staticmethod
    def _load_csv(path: str):
        with open(path, 'r') as traj_file:

            def check_empty_val(val, cast=True):
                if val == '':
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

                for j in range(dim):
                    r[j].append(float(row[j]))

                for j, k in enumerate(range(dim, dim + ang_dim)):
                    ang[j] = row[k]

                t.append(float(row[-1]))

            if not ang:
                ang = None

            return Trajectory(dimensions=r, t=t, ang=ang, dt=dt,
                              traj_id=traj_id)

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
            raise ValueError('Path does not exist.')
        if not path.is_file():
            raise ValueError("Path must be a file.")

        file_type = path.suffix

        try:
            if file_type == '.json':
                return Trajectory._load_json(file_path)
            elif file_type == '.csv':
                return Trajectory._load_csv(file_path)
            else:
                raise ValueError("Invalid file type.")
        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as exc:
            raise LoadTrajectoryError(path) from exc

    @staticmethod
    def load_folder(folder_path='.', recursively: bool = False):
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
                    print(f'Ignoring: {load_exception.path}')
            if not recursively:
                break
        return trajectories
