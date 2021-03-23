import numpy as np
from yupi.analyzing.transformations import wrap_theta

def turning_angles(trajectory, accumulate=False, degrees=False, centered=False):
    dx = trajectory.x_diff()
    dy = trajectory.y_diff()
    theta = np.arctan2(dy, dx)

    if not accumulate:
        theta = np.ediff1d(theta)  # relative turning angles
    else:
        theta -= theta[0]          # cumulative turning angles

    return wrap_theta(theta, degrees, centered)