import numpy as np
from typing import NamedTuple

TrajectoryPoint = NamedTuple('TrajectoryPoint', x=float, y=float, z=float,
                             t=float, theta=float)

class Trajectory():

    def __init__(self, x_arr: np.ndarray, y_arr: np.ndarray,
                 z_arr: np.ndarray = None, t_arr: np.ndarray = None,
                 theta_arr: np.ndarray = None, dt: float = None):

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
        self.dt = dt

    def __len__(self):
        return len(self.x_arr)

    def __iter__(self):
        for i in range(len(self)):

            x = self.x_arr[i]
            y = self.y_arr[i]     

            y, z, t, theta, = None, None, None, None

            if not self.y_arr is None:
                y = self.y_arr[i]
            if not self.z_arr is None:
                z = self.z_arr[i]
            if not self.time_arr is None:
                t = self.t_arr[i]
            if not self.theta_arr is None:
                theta = self.theta_arr[i]
            yield TrajectoryPoint(x, y, z, t, theta)
