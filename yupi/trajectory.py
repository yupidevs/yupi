import numpy as np
import json
import csv
import os
from typing import NamedTuple
from pathlib import Path

TrajectoryPoint = NamedTuple('TrajectoryPoint', x=float, y=float, z=float,
                             t=float, theta=float)


class Trajectory():
    """
    A Trajectory object represents a multidimensional trajectory. 
    It can be iterated to obtain the corresponding point for each timestep.

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
    dt : float
        If no time data (``t``) is given this represents the time
        between each position data value.
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

    def __init__(self, x: np.ndarray, y: np.ndarray = None,
                 z: np.ndarray = None, t: np.ndarray = None,
                 theta: np.ndarray = None, dt: float = 1.0, 
                 id: str = None):

        self.data = [x, y, z, t, theta]        


        for i, item in enumerate(self.data):
            if item is not None:
                self.data[i] = np.array(item)

        lengths = [len(item) for item in self.data if item is not None]
        
        if x is None:
            raise ValueError('Trajectory requires at least one dimension')
        elif lengths.count(lengths[0]) != len(lengths):
            raise ValueError('All input arrays must have the same shape')

        self.dt = dt
        self.id = id
    
    @property
    def x(self) -> np.ndarray:
        """np.ndarray : Array containing position data of X axis."""
        return self.data[0]

    @property
    def y(self) -> np.ndarray:
        """np.ndarray : Array containing position data of Y axis."""
        return self.data[1]

    @property
    def z(self) -> np.ndarray:
        """np.ndarray : Array containing position data of Z axis."""
        return self.data[2]

    @property
    def t(self) -> np.ndarray:
        """np.ndarray : Array containing time data."""
        return self.data[3]

    @property
    def theta(self) -> np.ndarray:
        """np.ndarray : Array containing angle data."""
        return self.data[4]
    
    @property
    def dim(self) -> int:
        """int : Trajectory spacial dimensions (can be 1, 2 or 3)."""
        for i, d in enumerate(self.data[:3]):
            if d is None:
                return i
        return 3
        
    def __len__(self):
        return len(self.x)

    def __iter__(self):
        current_time = 0
        for i in range(len(self)):
            # x, y, z, t, theta
            sp = [None]*5

            for j, d in enumerate(self.data):
                sp[j] = d[i] if d is not None else None

            if sp[3] is None and self.dt is not None: 
                sp[3] = current_time
                current_time += self.dt

            x, y, z, t, theta = sp
            yield TrajectoryPoint(x, y, z, t, theta)
                                  
    def t_diff(self):
        """
        Estimates the time difference between each couple 
        of consecutive samples in the Trajectory.

        Returns
        ----------
        t_diff : np.ndarray
            Array containing the time difference between consecutive samples.
        """

        if self.t is not None:
            return np.ediff1d(self.t)

    def x_diff(self):
        """
        Estimates the spacial difference between each couple 
        of consecutive samples in the x-axis of the Trajectory.

        Returns
        ----------
        x_diff : np.ndarray
            Array containing the x-axis difference between consecutive samples.
        """

        return np.ediff1d(self.x)

    def y_diff(self):
        """
        Estimates the spacial difference between each couple 
        of consecutive samples in the y-axis of the Trajectory.

        Returns
        ----------
        y_diff : np.ndarray
            Array containing the y-axis difference between consecutive samples.
        None:
            If Trajectory dim is less than 2
        """

        if self.y is not None:
            return np.ediff1d(self.y)

    def z_diff(self):
        """
        Estimates the spacial difference between each couple 
        of consecutive samples in the z-axis of the Trajectory.

        Returns
        ----------
        z_diff : np.ndarray
            Array containing the z-axis difference between consecutive samples.
        None:
            If Trajectory dim is less than 3
        """

        if self.z is not None:
            return np.ediff1d(self.z)

    def theta_diff(self):
        """
        Estimates the spacial difference between each couple 
        of consecutive samples in the theta array of the Trajectory.

        Returns
        ----------
        theta_diff : np.ndarray
            Array containing the theta array difference between consecutive samples.
        None:
            If Trajectory doesn't have theta informantion
        """

        if self.theta is not None:
            return np.ediff1d(self.theta)

    def diff(self):
        """
        Estimates the spacial difference between each couple 
        of consecutive samples across all the spacial dimensions 
        of the Trajectory object.

        Returns
        ----------
        diff : np.ndarray
            Array containing the difference between consecutive samples.
        """
        dx = self.x_diff()
        if self.dim == 1:
            return dx
        dy = self.y_diff()
        if self.dim == 2:
            return np.sqrt(dx**2 + dy**2)
        else:
            dz = self.y_diff()
            return np.sqrt(dx**2 + dy**2 + dz**2)
            

    def x_velocity(self):
        """
        Computes the velocity in the x-axis of the Trajectory.

        Returns
        ----------
        x_vel : np.ndarray
            Array containing the x-axis velocity of the Trajectory.
        """

        return self.x_diff() / self.dt

    def y_velocity(self):
        """
        Computes the velocity in the y-axis of the Trajectory.

        Returns
        ----------
        y_vel : np.ndarray
            Array containing the y-axis velocity of the Trajectory.
        None:
            If Trajectory dim is less than 2
        """

        if self.y is not None:
            return self.y_diff() / self.dt

    def z_velocity(self):
        """
        Computes the velocity in the z-axis of the Trajectory.

        Returns
        ----------
        z_vel : np.ndarray
            Array containing the z-axis velocity of the Trajectory.
        None:
            If Trajectory dim is less than 3
        """

        if self.z is not None:
            return self.z_diff() / self.dt

    def theta_velocity(self):
        """
        Computes the angular velocity from the theta array of the Trajectory.

        Returns
        ----------
        omega : np.ndarray
            Array containing the angular velocity of the Trajectory.
        None:
            If Trajectory doesn't have theta informantion
        """

        if self.theta is not None:
            return self.theta_diff() / self.dt

    def velocity(self):
        """
        Estimates the velocity across all the spacial dimensions 
        of the Trajectory object.

        Returns
        ----------
        v : np.ndarray
            Array containing the velocity of the Trajectory.
        """

        return self.diff() / self.dt

    def position_vectors(self):
        """
        Fetch the spacial components across all axis.

        Returns
        ----------
        r : np.ndarray
            Array containing all the position arrays of the Trajectory.
        """

        # get the components of the position
        r = self.data[:self.dim]

        # transpose to have time/dimension as first/second axis
        r = np.transpose(r)
        return r

    def velocity_vectors(self):
        """
        Returns the velocity components across all axis.

        Returns
        ----------
        v : np.ndarray
            Array containing all the velocity arrays of the Trajectory.
        """

        v = []
        
        # append velocity x-component
        v.append(self.x_velocity())

        # append velocity y-component
        if self.dim >= 2:
            v.append(self.y_velocity())

        # append velocity z-component
        if self.dim == 3:
            v.append(self.z_velocity())

        # transpose to have time/dimension as first/second axis
        v = np.transpose(v)
        return v

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
            'x' : convert_to_list(self.x),
            'y' : convert_to_list(self.y),
            'z' : convert_to_list(self.z),
            't' : convert_to_list(self.t),
            'theta' : convert_to_list(self.theta)
        }
        with open(str(path), 'w') as f:
            json.dump(json_dict, f)

    def _save_csv(self, path):
        with open(str(path), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([self.id, self.dt])
            for tp in self:
                writer.writerow([tp.x, tp.y, tp.z, tp.t, tp.theta])

    def save(self, file_name: str, path: str = '.', file_type: str = 'json',
               overwrite: bool = True):
        """
        Saves the trajectory to disk.

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
            is True).

        Examples
        --------
        >>> t = Trajectory(x=[0.37, 1.24, 1.5]) 
        >>> t.save('my_track')

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
            Path where to save all the trajectory. (Default is ``'.'``).
        file_type : str
            Type of the file. (Default is ``json``).

            The only types avaliable are: ``json`` and ``csv``.
        overwrite : bool
            Wheter or not to overwrite the file if it already exists. (Default
            is True).

        Examples
        --------
        >>> trajectories = [
            Trajectory(x=[0.37, 1.24, 1.5]), 
            Trajectory(x=[1, 2], y=[3, 4])] 
        >>> Trajectory.save_trajectories(trajectories)
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
            x, y, z, t = data['x'], data['y'], data['z'], data['t']
            theta = data['theta']
            return Trajectory(x=x, y=y, z=z, t=t, theta=theta, dt=dt, 
                              id=traj_id)

    def _load_csv(path: str):
        with open(path, 'r') as f:

            def check_empty_val(val, cast=True):
                if val == '':
                    return None
                return float(val) if cast else val     

            # x, y, z, t, theta
            dat = [[], [], [], [], []]
            traj_id, dt = None, None
                
            for i, row in enumerate(csv.reader(f)):
                if i == 0:
                    traj_id = check_empty_val(row[0], cast=False)
                    dt = check_empty_val(row[1])
                    continue

                for j in range(len(dat)):
                    dat[j].append(check_empty_val(row[j]))
            
            for i, d in enumerate(dat):
                if any([item is None for item in d]):
                    dat[i] = None

            x, y, z, t, theta = dat
            return Trajectory(x=x, y=y, z=z, t=t, theta=theta, dt=dt,
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

if __name__ == '__main__':

    traj_1 = Trajectory(
        x=[1.0, 2.0],
        y=[2.0, 3.0]
    )

    traj_2 = Trajectory(
        x=[10.0, 20.0],
        y=[20.0, 30.0]
    )

    tps = [(tp.x, tp.y) for tp in traj_1]
    assert tps == [(1,2),(2,3)]

    Trajectory.save_trajectories([traj_1, traj_2], file_type='csv')

    trajs = Trajectory.load_folder()
    
    t1 = trajs[0]

    assert t1.x[0] == 1.0
    assert t1.x[1] == 2.0


    