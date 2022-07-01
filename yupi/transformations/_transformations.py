from typing import Optional, Tuple

import numpy as np

from yupi import Trajectory
from yupi.transformations._affine_estimator import _affine_matrix


def _affine2camera(
    theta: np.ndarray, t_x: np.ndarray, t_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    x_cam2lab, y_cam2lab, theta_cam2lab = np.zeros((3, theta.shape[0] + 1))
    theta_cam2lab[1:] = np.cumsum(theta)

    for i in range(theta.size):
        affine_mat = _affine_matrix(
            theta_cam2lab[i + 1], x_cam2lab[i], y_cam2lab[i], inverse=True
        )
        shift_vec = [-t_x[i], -t_y[i], 1]
        x_cam2lab[i + 1], y_cam2lab[i + 1] = affine_mat @ shift_vec

    return x_cam2lab[1:], y_cam2lab[1:], theta_cam2lab[1:]


def _camera2obj(
    x_obj2cam: np.ndarray,
    y_obj2cam: np.ndarray,
    x_cam2lab: np.ndarray,
    y_cam2lab: np.ndarray,
    theta_cam2lab: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    x_obj2lab, y_obj2lab = np.empty((2, x_obj2cam.size))

    for i in range(x_obj2cam.size):
        affine_mat = _affine_matrix(
            theta_cam2lab[i], x_cam2lab[i], y_cam2lab[i], inverse=True
        )
        shift_vec = [x_obj2cam[i], y_obj2cam[i], 1]
        x_obj2lab[i], y_obj2lab[i] = affine_mat @ shift_vec

    return x_obj2lab, y_obj2lab


def _affine2obj(
    theta: np.ndarray,
    t_x: np.ndarray,
    t_y: np.ndarray,
    x_obj2cam: np.ndarray,
    y_obj2cam: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    x_cam2lab, y_cam2lab, theta_cam2lab = _affine2camera(theta, t_x, t_y)
    x_obj2lab, y_obj2lab = _camera2obj(
        x_obj2cam, y_obj2cam, x_cam2lab, y_cam2lab, theta_cam2lab
    )
    return x_obj2lab, y_obj2lab


def add_moving_FoR(  # pylint: disable=invalid-name
    traj: Trajectory,
    reference: Tuple[np.ndarray, np.ndarray, np.ndarray],
    start_at_origin: bool = True,
    new_traj_id: Optional[str] = None,
) -> Trajectory:
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

    theta, t_x, t_y = reference
    x_al, y_al = _affine2obj(theta, t_x, t_y, traj.r.x, traj.r.y)

    if start_at_origin:
        x_al = x_al - x_al[0]
        y_al = y_al - y_al[0]

    moved_traj = Trajectory(
        x=x_al,
        y=y_al,
        t=traj.t,
        traj_id=new_traj_id,
        diff_est=traj.diff_est,
    )
    return moved_traj
