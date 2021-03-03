import numpy as np
from typing import NamedTuple

# TODO: check fields names
TrajectoryPoint = NamedTuple('TrajectoryPoint', x=float, y=float, z=float,
                             tm=float, th=float)

class Trajectory():

    def __init__(self,
                 x_arr: np.ndarray,
                 y_arr: np.ndarray,
                 z_arr: np.ndarray = None,
                 time_arr: np.ndarray = None,
                 theta_arr: np.ndarray = None,
                 dt: float = None):

        if len(x_arr) != len(y_arr):
            # TODO: write error message
            raise ValueError()

        self.x_arr = x_arr
        self.y_arr = y_arr
        self.z_arr = z_arr
        self.time_arr = time_arr
        self.theta_arr = theta_arr
        self.dt = dt

    def __len__(self):
        return len(self.x_arr)

    def __iter__(self):
        for i in range(len(self)):

            x = self.x_arr[i]
            y = self.y_arr[i]     

            z, tm, th, = None, None, None

            if self.z_arr:
                z = self.z_arr[i]
            if self.time_arr:
                tm = self.time_arr[i]
            if self.theta_arr:
                th = self.theta_arr[i]
            yield TrajectoryPoint(x, y, z, tm, th)
