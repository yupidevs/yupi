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
    x_arr : np.ndarray
        Array containing position data of X axis.
    y_arr : np.ndarray
        Array containing position data of Y axis. (Default is None).
    z_arr : np.ndarray
        Array containing position data of X axis. (Default is None).
    t_arr : np.ndarray
        Array containing time data. (Default is None).
    theta_arr : np.ndarray
        Array containing angle data. (Default is None).
    dt : float
        If no time data (``t_arr``) is given this represents the time
        between each position data value.
    id : str
        Id of the trajectory.

    Attributes
    ----------
    x_arr : np.ndarray
        Array containing position data of X axis.
    y_arr : np.ndarray
        Array containing position data of Y axis.
    z_arr : np.ndarray
        Array containing position data of X axis.
    t_arr : np.ndarray
        Array containing time data.
    theta_arr : np.ndarray
        Array containing angle data.
    dt : float
        If no time data (``t_arr``) is given this represents the time
        between each position data value.
    id : str
        Id of the trajectory.

    Raises
    ------
    ValueError
        If ``x_arr`` is not given.
    ValueError
        If all the given position data (``x_arr``, ``y_arr`` and/or ``z_arr``)
        does not have the same shape.
    """

    def __init__(self, x_arr: np.ndarray, y_arr: np.ndarray = None,
                 z_arr: np.ndarray = None, t_arr: np.ndarray = None,
                 theta_arr: np.ndarray = None, dt: float = None, 
                 id: str = None):

        if x_arr is None:
            raise ValueError('Trajectory requires at least one dimension')
        elif y_arr is not None and z_arr is None:
            if len(x_arr) != len(y_arr):
                raise ValueError('X and Y arrays must have the same shape')
        elif y_arr is not None and z_arr is not None:
            if len(x_arr) != len(y_arr) != len(z_arr):
                raise ValueError('X and Z arrays must have the same shape')
        if t_arr is not None:
            if len(x_arr) != len(t_arr):
                raise ValueError('X and Time arrays must have the same shape')
        if theta_arr is not None:
            if len(x_arr) != len(theta_arr):
                raise ValueError('X and Theta arrays must have the same shape')

        self.x_arr = x_arr
        self.y_arr = y_arr
        self.z_arr = z_arr
        self.t_arr = t_arr
        self.theta_arr = theta_arr

        if self.x_arr is not None:
            self.x_arr = np.array(x_arr)
        if self.y_arr is not None:
            self.y_arr = np.array(y_arr)
        if self.z_arr is not None:
            self.z_arr = np.array(z_arr)
        if self.t_arr is not None:
            self.t_arr = np.array(t_arr)
        if self.theta_arr is not None:
            self.theta_arr = np.array(theta_arr)

        self.dt = dt
        self.id = id
    
    def __len__(self):
        return len(self.x_arr)

    def __iter__(self):
        current_time = 0
        for i in range(len(self)):

            x = self.x_arr[i]
            y = self.y_arr[i]     

            y, z, t, theta, = None, None, None, None

            if self.y_arr is not None:
                y = self.y_arr[i]
            if self.z_arr is not None:
                z = self.z_arr[i]
            if self.t_arr is not None:
                t = self.t_arr[i]
            elif self.dt is not None:
                t = current_time
                current_time += self.dt
            if self.theta_arr is not None:
                theta = self.theta_arr[i]

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
                'x_arr' : convert_to_list(self.x_arr),
                'y_arr' : convert_to_list(self.y_arr),
                'z_arr' : convert_to_list(self.z_arr),
                't_arr' : convert_to_list(self.t_arr),
                'theta_arr' : convert_to_list(self.theta_arr)
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
                x_arr = data['x_arr']
                y_arr = data['y_arr']
                z_arr = data['z_arr']
                t_arr = data['t_arr']
                theta_arr = data['theta_arr']
                return Trajectory(x_arr=x_arr, y_arr=y_arr, z_arr=z_arr,
                                  t_arr=t_arr, theta_arr=theta_arr, dt=dt,
                                  id=traj_id)

            elif file_type == '.csv':

                def check_empty_val(val):
                    return None if val == '' else val               

                x_arr, y_arr, z_arr = [], [], []
                t_arr, theta_arr = [], []
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

                    add_val(x_arr, check_empty_val(row[0]))
                    add_val(y_arr, check_empty_val(row[1]))
                    add_val(z_arr, check_empty_val(row[2]))
                    add_val(t_arr, check_empty_val(row[3]))
                    add_val(theta_arr, check_empty_val(row[4]))
                
                x_arr = None if not x_arr else x_arr
                y_arr = None if not y_arr else y_arr
                z_arr = None if not z_arr else z_arr
                t_arr = None if not t_arr else t_arr
                theta_arr = None if not theta_arr else theta_arr

                return Trajectory(x_arr=x_arr, y_arr=y_arr, z_arr=z_arr,
                                  t_arr=t_arr, theta_arr=theta_arr, dt=dt,
                                  id=traj_id)
