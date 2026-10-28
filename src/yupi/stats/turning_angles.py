from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.projections import PolarAxes

from yupi._checkers import check_exact_dim, check_same_dt, check_uniform_time_spaced
from yupi.exceptions import TrajectoryError
from yupi.graphics._style import LIGHT_BLUE
from yupi.trajectory import Trajectory


def turning_angles(
    traj: Trajectory,
    accumulate: bool = False,
    degrees: bool = False,
    centered: bool = False,
    wrap: bool = True,
) -> np.ndarray:
    """
    Return the sequence of turning angles that forms the trajectory.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory.
    accumulate : bool, optional
        If True, turning angles are measured with respect to an axis
        defined by the initial velocity (i.e., angles between initial
        and current velocity). Otherwise, relative turning angles
        are computed (i.e., angles between succesive velocity
        vectors). By default False.
    degrees : bool, optional
        If True, angles are given in degrees. Otherwise, the units
        are radians. By default False.
    centered : bool, optional
        If True, angles are wrapped on the interval ``[-pi, pi]``.
        Otherwise, the interval ``[0, 2*pi]`` is chosen. By default
        False.
    wrap : bool, optional
        If True, angles are wrapped in a certain interval (depending
        on ``centered`` param). By default True.

    Returns
    -------
    np.ndarray
        Turning angles where each position in the array correspond
        to a given time instant.
    """
    if traj.dim != 2:
        raise TrajectoryError(
            traj, "The trajectory must be 2D to estimate turning angles"
        )

    d_r = traj.delta_r
    d_x, d_y = d_r.x, d_r.y
    theta = np.arctan2(d_y, d_x)

    if not accumulate:
        theta = np.ediff1d(theta)  # Relative turning angles
    else:
        theta -= theta[0]  # Accumulative turning angles

    if degrees:
        theta = np.rad2deg(theta)

    if not wrap:
        return theta

    discont = 360 if degrees else 2 * np.pi
    if not centered:
        return theta % discont

    discont_half = discont / 2
    return -((discont_half - theta) % discont - discont_half)


class TurningAngleStat:
    """
    Estimate all the turning angles that forms a set of trajectories.

    Parameters
    ----------
    trajs : list[Trajectory]
        Input list of trajectories.
    accumulate : bool, optional
        If True, turning angles are measured with respect to an axis
        define by the initial velocity (i.e., angles between initial
        and current velocity). Otherwise, relative turning angles
        are computed (i.e., angles between succesive velocity vectors).
        By default False.
    degrees : bool, optional
        If True, angles are given in degrees. Otherwise, the units
        are radians. By default False.
    centered : bool, optional
        If True, angles are wrapped on the interval ``[-pi, pi]``.
        Otherwise, the interval ``[0, 2*pi]`` is chosen. By default
        False.
    wrap : bool, optional
        If True, angles are wrapped in a certain interval (depending
        on ``centered`` param). By default True.

    Attributes
    ----------
    theta : np.ndarray
        Concatenated array of turning angles for a list of Trajectory
        objects.
    """

    def __init__(
        self,
        trajs: list[Trajectory],
        accumulate: bool = False,
        degrees: bool = False,
        centered: bool = False,
        wrap: bool = True,
    ) -> None:
        check_exact_dim(trajs, 2)
        check_same_dt(trajs)
        check_uniform_time_spaced(trajs)

        self.trajs = trajs
        self.__theta: np.ndarray | None = None
        self.accumulate = accumulate
        self.degrees = degrees
        self.centered = centered
        self.wrap = wrap

    @property
    def theta(self) -> np.ndarray:
        """
        Get the turning angles of the trajectories.

        Returns
        -------
        list[list[float]]
            List of turning angles for each trajectory.
        """
        if self.__theta is None:
            self.__theta = np.concatenate(
                [
                    turning_angles(
                        traj,
                        accumulate=self.accumulate,
                        degrees=self.degrees,
                        centered=self.centered,
                        wrap=self.wrap,
                    )
                    for traj in self.trajs
                ]
            )
        return self.__theta

    def plot(
        self,
        bins: int = 36,
        show: bool = True,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot a histogram of the array of angles ``ang``.

        Parameters
        ----------
        ang : np.ndarray
            Array of angles.
        bins: int
            Number of histogram bins.
        show : bool, optional
            If True, the plot is shown. By default True.
        ax : Axes, optional
            Axes to plot. By default None.

        Returns
        -------
        PolarAxes
            Axes of the plot.

        Raises
        ------
        ValueError
            If the axes is not polar.
        """

        if ax is None:
            ax = plt.axes(projection="polar")

        if not isinstance(ax, PolarAxes):
            raise ValueError("The axes must be polar")

        default_kwargs: Any = {
            "color": LIGHT_BLUE,
            "ec": (0, 0, 0, 0.6),
            "density": True,
        }
        default_kwargs.update(kwargs)
        plt.hist(self.theta, bins, **default_kwargs)
        ax.set_theta_zero_location("N")
        ax.set_rlabel_position(135)
        ax.set_axisbelow(True)
        plt.xlabel("turning angles pdf")
        if show:
            plt.show()

        return ax
