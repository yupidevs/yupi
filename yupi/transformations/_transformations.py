from typing import Tuple
import numpy as np
from yupi import Trajectory
from yupi.transformations._affine_estimator import _affine_matrix


def add_moving_FoR(traj: Trajectory,
                   reference: Tuple[np.ndarray, np.ndarray, np.ndarray],
                   start_at_origin: bool = True, new_traj_id: str = None):
    """
    This function fuses the information of a trajectory with an
    external reference of the motion of the Frame of Reference
    (FoR).

    It allows to remap the information gathered in local SoRs
    to a more general FoR.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory.
    reference : Tuple[np.ndarray,np.ndarray,np.ndarray]
        Angular and translational parameters of the form
        ``(ang:np.ndarray, tx:np.ndarray, ty:np.ndarray)`` that
        accounts for the orientation and displacement of the reference.
    start_at_origin : bool, optional
        If True, set initial position at the origin. By default True.

    Returns
    -------
    Trajectory
        Output trajectory in the lab frame of reference.
    """

    def affine2camera(theta, tx, ty):
        x_cl, y_cl, theta_cl = np.zeros((3, theta.size + 1))
        theta_cl[1:] = np.cumsum(theta)

        for i in range(theta.size):
            A = _affine_matrix(theta_cl[i + 1], x_cl[i], y_cl[i], R_inv=True)
            x_cl[i + 1], y_cl[i + 1] = A @ [-tx[i], -ty[i], 1]

        x_cl, y_cl, theta_cl = x_cl[1:], y_cl[1:], theta_cl[1:]
        return x_cl, y_cl, theta_cl

    def camera2obj(x_ac, y_ac, x_cl, y_cl, theta_cl):
        x_al, y_al = np.empty((2, x_ac.size))

        for i in range(x_ac.size):
            A = _affine_matrix(theta_cl[i], x_cl[i], y_cl[i], R_inv=True)
            x_al[i], y_al[i] = A @ [x_ac[i], y_ac[i], 1]

        return x_al, y_al

    def affine2obj(theta, tx, ty, x_ac, y_ac):
        x_cl, y_cl, theta_cl = affine2camera(theta, tx, ty)
        x_al, y_al = camera2obj(x_ac, y_ac, x_cl, y_cl, theta_cl)
        return x_al, y_al

    theta, tx, ty = reference

    x_al, y_al = affine2obj(theta, tx, ty, traj.r.x, traj.r.y)

    if start_at_origin:
        x_al = x_al - x_al[0]
        y_al = y_al - y_al[0]

    traj.x = x_al
    traj.y = y_al

    moved_traj = Trajectory(
        x=x_al,
        y=y_al,
        ang=traj.ang,
        t=traj.t,
        traj_id=new_traj_id
    )
    return moved_traj
