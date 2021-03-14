import numpy as np
import json
import csv
from typing import NamedTuple
from pathlib import Path

TrajectoryPoint = NamedTuple('TrajectoryPoint', x=float, y=float, z=float,
                             t=float, theta=float)


class Trajectory():
    """
    Represents a trajectory.

    Parameters
    ----------
    x : np.ndarray
        Array containing position data of X axis.
    y : np.ndarray
        Array containing position data of Y axis. (Default is None).
    z : np.ndarray
        Array containing position data of X axis. (Default is None).
    t : np.ndarray
        Array containing time data. (Default is None).
    theta : np.ndarray
        Array containing angle data. (Default is None).
    dt : float
        If no time data (``t``) is given this represents the time
        between each position data value.
    id : str
        Id of the trajectory.

    Attributes
    ----------
    x : np.ndarray
        Array containing position data of X axis.
    y : np.ndarray
        Array containing position data of Y axis.
    z : np.ndarray
        Array containing position data of X axis.
    t : np.ndarray
        Array containing time data.
    theta : np.ndarray
        Array containing angle data.
    dt : float
        If no time data (``t``) is given this represents the time
        between each position data value.
    id : str
        Id of the trajectory.

    Raises
    ------
    ValueError
        If ``x`` is not given.
    ValueError
        If all the given position data (``x``, ``y`` and/or ``z``)
        does not have the same shape.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray = None,
                 z: np.ndarray = None, t: np.ndarray = None,
                 theta: np.ndarray = None, dt: float = None, 
                 id: str = None):

        if x is None:
            raise ValueError('Trajectory requires at least one dimension')
        elif y is not None and z is None:
            if len(x) != len(y):
                raise ValueError('X and Y arrays must have the same shape')
        elif y is not None and z is not None:
            if len(x) != len(y) != len(z):
                raise ValueError('X and Z arrays must have the same shape')
        if t is not None:
            if len(x) != len(t):
                raise ValueError('X and Time arrays must have the same shape')
        if theta is not None:
            if len(x) != len(theta):
                raise ValueError('X and Theta arrays must have the same shape')

        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.theta = theta

        if self.x is not None:
            self.x = np.array(x)
        if self.y is not None:
            self.y = np.array(y)
        if self.z is not None:
            self.z = np.array(z)
        if self.t is not None:
            self.t = np.array(t)
        if self.theta is not None:
            self.theta = np.array(theta)

        self.dt = dt
        self.id = id
    
    def __len__(self):
        return len(self.x)

    def __iter__(self):
        current_time = 0
        for i in range(len(self)):

            x = self.x[i]
            y = self.y[i]     

            y, z, t, theta, = None, None, None, None

            if self.y is not None:
                y = self.y[i]
            if self.z is not None:
                z = self.z[i]
            if self.t is not None:
                t = self.t[i]
            elif self.dt is not None:
                t = current_time
                current_time += self.dt
            if self.theta is not None:
                theta = self.theta[i]

            yield TrajectoryPoint(x, y, z, t, theta)

    def save(self, file_name: str, path: str = '.', file_type: str = 'json',
               overwrite: bool = True):
        """
        Saves a trajectory

        Parameters
        ----------
        file_name : str
            Name of the file.
        path : str
            Path where to save the trajectory. (Default is ``'.'``).
        file_time : str
            Type of the file. (Default is ``json``).

            The only types avaliable are: ``json`` and ``csv``.
        overwrite : bool
            Wheter or not to overwrite the file if it already exists. (Default
            is True)

        Raises
        ------        
        ValueError
            If ``override`` parameter is ``False`` and the file already exists.
        ValueError
            If ``file_type`` is not ``json`` or ``csv``.
        """

        # Contruct full path
        full_path = Path(path) / Path(f'{file_name}.{file_type}')

        # Check file existance
        if not overwrite and full_path.exists():
            raise ValueError(f"File '{str(full_path)}' already exist")

        def convert_to_list(array_data):
            if array_data is None:
                return array_data
            if array_data is not list:
                array_data = list(array_data)
            return array_data

        if file_type == 'json':
            json_dict = {
                'dt' : self.dt,
                'id' : self.id,
                'x' : convert_to_list(self.x),
                'y' : convert_to_list(self.y),
                'z' : convert_to_list(self.z),
                't' : convert_to_list(self.t),
                'theta' : convert_to_list(self.theta)
            }
            with open(str(full_path), 'w') as f:
                json.dump(json_dict, f)

        elif file_type == 'csv':
            with open(str(full_path), 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([self.id, self.dt])
                for tp in self:
                    writer.writerow([tp.x, tp.y, tp.z, tp.t, tp.theta])
        else:
            raise ValueError(f"Invalid export file type '{file_type}'")

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
        Trajecotry
            Trajectory loaded.

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
    
        # Check valid file type
        file_type = path.suffix
        if not path.suffix in ['.json', '.csv']:
            raise ValueError("Invalid file type.")

        with open(file_path, 'r') as f:
            if file_type == '.json':

                data = json.load(f)
                dt = data['dt']
                traj_id = data['id']
                x = data['x']
                y = data['y']
                z = data['z']
                t = data['t']
                theta = data['theta']
                return Trajectory(x=x, y=y, z=z,
                                  t=t, theta=theta, dt=dt,
                                  id=traj_id)

            elif file_type == '.csv':

                def check_empty_val(val):
                    return None if val == '' else val               

                x, y, z = [], [], []
                t, theta = [], []
                traj_id, dt = None, None

                def add_val(arr, val):
                    if arr is not None:
                        arr.append(val)
                    
                for i, row in enumerate(csv.reader(f)):
                    if i == 0:
                        traj_id = check_empty_val(row[0])
                        dt = check_empty_val(row[1])
                        if dt is not None:
                            dt = float(dt)
                        continue

                    add_val(x, check_empty_val(row[0]))
                    add_val(y, check_empty_val(row[1]))
                    add_val(z, check_empty_val(row[2]))
                    add_val(t, check_empty_val(row[3]))
                    add_val(theta, check_empty_val(row[4]))
                
                x = None if not x else x
                y = None if not y else y
                z = None if not z else z
                t = None if not t else t
                theta = None if not theta else theta

                return Trajectory(x=x, y=y, z=z,
                                  t=t, theta=theta, dt=dt,
                                  id=traj_id)


    # create a copy instance that is sampled at step increments
    def sample_traj(self, step=1, step_in_seconds=False):
        if step_in_seconds:
            step = int(step / self.dt)
        x = self.x[::step]
        y = self.y[::step]
        traj = Trajectory(x, y, dt=step*self.dt)
        return traj

    
    # set jump length / velocity attributes
    def get_jumps(self, v=False):
        dx = np.ediff1d(self.x)
        dy = np.ediff1d(self.y)
        dr = np.sqrt(dx**2 + dy**2)
        jumps = np.array([dx, dy, dr])

        if not v:
            attr = ['dx', 'dy', 'dr']
        else:
            attr = ['vx', 'vy', 'v']
            jumps /= self.dt

        for i in range(3):
            setattr(self, attr[i], jumps[i])
        
        return self