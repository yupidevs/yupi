import json
import csv
import os
import numpy as np
from typing import NamedTuple
from pathlib import Path
from numpy.linalg.linalg import norm as nrm


class Vector(np.ndarray):

    @property
    def norm(self):
        return Vector.create([nrm(p) for p in self])

    @property
    def delta(self):
        if len(self.shape) > 1:
            new_vec = []
            for i in range(self.shape[1]):
                new_vec.append(np.ediff1d(self[:,i]))
            return Vector.create(new_vec).T
        else:
            return np.ediff1d(self)

    @property
    def x(self):
        return self.component(0)

    @property
    def y(self):
        return self.component(1)

    @property
    def z(self):
        return self.component(2)

    def component(self, dim):
        if len(self.shape) < 2:
            raise TypeError('Operation not supperted on simple vectors')
        if not isinstance(dim, int):
            raise TypeError("Parameter 'dim' must be an integer")
        if self.shape[1] < dim + 1:
            raise ValueError(f'Vector has not component {dim}')
        return self[:,dim]

    @staticmethod
    def create(*args, **kwargs):
        arr = np.array(*args, **kwargs)
        return arr.view(Vector)

class Trajectory():
    """
    A Trajectory object represents a multidimensional trajectory. 
    It can be iterated to obtain the corresponding point for each timestep.

    Parameters
    ----------
    x : np.ndarray
        Array containing position data of X axis.
    y : np.ndarray
        Array containing position data of Y axis, by default None.
    z : np.ndarray
        Array containing position data of X axis, by default None.
    t : np.ndarray
        Array containing time data, by default None.
    theta : np.ndarray
        Array containing angle data, by default None.
    dt : float
        If no time data is given this represents the time between each
        position data value.
    id : str
        Id of the trajectory.

    Attributes
    ----------
    dt : float
        If no time data is given this represents the time between each
        position data value.
    id : str
        Id of the trajectory.

    Examples
    --------
    You can create a trajectory object by giving the arrays that represent it:

    >>> x = [0, 1.0, 0.63, -0.37, -1.24, -1.5, -1.08, -0.19, 0.82, 1.63, 1.99, 1.85]
    >>> y = [0, 0, 0.98, 1.24, 0.69, -0.3, -1.23, -1.72, -1.63, -1.01, -0.06, 0.94]
    >>> Trajectory(x=x, y=y, id="Spiral")


    Raises
    ------
    ValueError
        If ``x`` is not given.
    ValueError
        If all the given input data (``x``, ``y``, ``z``, ``t``, ``theta``)
        does not have the same shape.
    """

    def __init__(self, x: np.ndarray = None, y: np.ndarray = None,
                 z: np.ndarray = None, points: np.ndarray = None,
                 dimensions: np.ndarray = None, t: np.ndarray = None,
                 ang: np.ndarray = None, dt: float = 1.0,
                 id: str = None):

        from_xyz = x is not None
        from_points = points is not None
        from_dimensions = dimensions is not None

        if from_xyz + from_points + from_dimensions > 1:
            raise ValueError("Positional data must come only from one way: " \
                             "'xyz' data, 'points' data or 'dimensions' data.")

        self.r = None
        data = [t, ang]
        lengths = [len(item) for item in data if item is not None]

        for i, item in enumerate(data):
            if item is not None:
                data[i] = Vector.create(item)

        if from_xyz:
            dimensions = [d for d in [x,y,z] if d is not None]
            from_dimensions = True

        if from_dimensions:
            if len(dimensions) == 0:
                raise ValueError('Trajectory requires at least one dimension.')
            lengths.extend([len(d) for d in dimensions])
            self.r = Vector.create(dimensions).T

        if from_points:
            lengths.append(len(points))
            self.r = Vector.create(points)

        if lengths.count(lengths[0]) != len(lengths):
            raise ValueError('All input arrays must have the same shape.')

        if self.r is None:
            raise ValueError('No position data were given.')

        self.dt = dt
        self.id = id
        self.t = None if data[0] is None else Vector.create(data[0])
        self.ang = None if data[1] is None else Vector.create(data[1])
        self.v: Vector = self.r.delta / self.dt

    @property
    def x(self) -> np.ndarray:
        """np.ndarray : Array containing position data of X axis."""
        return self.r.x

    @property
    def y(self) -> np.ndarray:
        """np.ndarray : Array containing position data of Y axis."""
        return self.r.y

    @property
    def z(self) -> np.ndarray:
        """np.ndarray : Array containing position data of Z axis."""
        return self.r.z

    @property
    def dim(self) -> int:
        """int : Trajectory spacial dimensions."""
        return self.r.shape[1]

    def __len__(self):
        return self.r.shape[0]

    def __iter__(self):
        current_time = 0
        for i in range(len(self)):
            # *dim, t, theta
            data = list(self.r[i])
            data.extend([None,None])

            if self.t is not None:
                data[-2] = self.t[i]
            else:
                data[-2] = current_time
                current_time += self.dt

            if self.ang is not None:
                data[-1] = self.ang[i]

            yield data

    @property
    def delta_t(self):
        """
        Difference between each couple of consecutive samples in the
        Trajectory.

        Returns
        ----------
        np.ndarray
            Array containing the time difference between consecutive samples.
        """

        if self.t is not None:
            return self.t.delta

    @property
    def delta_r(self) -> Vector:
        """
        Difference between each couple of consecutive points in the
        Trajectory.

        Returns
        ----------
        np.ndarray
            Array containing the position difference between consecutive
            points.
        """

        return self.r.delta

    @property
    def delta_ang(self):
        """
        Difference between each couple of consecutive samples in the
        ``ang`` array of the Trajectory.

        Returns
        ----------
        np.ndarray
            Array containing the ``ang`` array difference between consecutive
            samples.
        None
            If Trajectory doesn't have ``ang`` informantion
        """

        if self.ang is not None:
            return self.ang.delta

    @property
    def ang_velocity(self):
        """
        Computes the angular velocity from the ang array of the Trajectory.

        Returns
        ----------
        np.ndarray
            Array containing the angular velocity of the Trajectory.
        None
            If Trajectory doesn't have ang informantion
        """

        if self.ang is not None:
            return self.ang.delta / self.dt

    def _save_json(self, path: str):
        def convert_to_list(array_data):
            if array_data is None:
                return array_data
            if array_data is not list:
                array_data = list(array_data)
            return array_data

        json_dict = {
            'dt' : self.dt,
            'id' : self.id,
            'dimensions' : { k:convert_to_list(v) 
                             for k,v in enumerate(self.r.T)},
            't' : convert_to_list(self.t),
            'theta' : convert_to_list(self.ang)
        }
        with open(str(path), 'w') as f:
            json.dump(json_dict, f)

    def _save_csv(self, path):
        with open(str(path), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([self.id, self.dt, self.dim])
            for tp in self:
                writer.writerow(tp)

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
            Wheter or not to overwrite the file if it already exists, by default
            True.

        Raises
        ------        
        ValueError
            If ``override`` parameter is ``False`` and the file already exists.
        ValueError
            If ``file_type`` is not ``json`` or ``csv``.

        Examples
        --------
        >>> t = Trajectory(x=[0.37, 1.24, 1.5]) 
        >>> t.save('my_track')
        """

        # Contruct full path
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
        Saves a list of trajectories to disk. Each Trajectory object will be saved
        in a separate file inside the given folder.

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
            Wheter or not to overwrite the file if it already exists, by default
            True.

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
        with open(path, 'r') as f:
            data = json.load(f)
            traj_id, dt = data['id'], data['dt']
            t = data['t']
            theta = data['theta']
            dims = [d for d in data['dimensions'].values()]

            return Trajectory(dimensions=dims, t=t, ang=theta, dt=dt, 
                              id=traj_id)

    @staticmethod
    def _load_csv(path: str):
        with open(path, 'r') as f:

            def check_empty_val(val, cast=True):
                if val == '':
                    return None
                return float(val) if cast else val

            # *dim, t, theta
            dat, t, ang = None, [], []
            traj_id, dt, dim = None, None, None

            for i, row in enumerate(csv.reader(f)):
                if i == 0:
                    traj_id = check_empty_val(row[0], cast=False)
                    dt = check_empty_val(row[1])
                    dim = int(row[2])
                    dat = [[] for _ in range(dim)]
                    continue

                for j in range(dim):
                    dat[j].append(check_empty_val(row[j]))

                t.append(check_empty_val(row[-2]))
                ang.append(check_empty_val(row[-1]))

            dat.extend([t, ang])

            for i, d in enumerate(dat):
                if any([item is None for item in d]):
                    dat[i] = None

            *dims, t, ang = dat
            return Trajectory(dimensions=dims, t=t, ang=ang, dt=dt,
                              id=traj_id)

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

        if file_type == '.json':
            return Trajectory._load_json(file_path)
        elif file_type == '.csv':
            return Trajectory._load_csv(file_path)
        else:
            raise ValueError("Invalid file type.")

    @staticmethod
    def load_folder(folder_path='.'):
        """
        Loads all the trajectories from a folder.

        Parameters
        ----------
        folder_path : str
            Path of the trajectories folder.

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
                except:  # TODO: add errors
                    pass
        return trajectories
