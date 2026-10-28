from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from yupi._checkers import check_same_dim
from yupi.graphics._style import LIGHT_YELLOW
from yupi.trajectory import Trajectory
from yupi.transformations import subsample


class SpeedStat:
    """
    Estimate speeds of the list of trajectories, ``trajs``,
    by computing displacements according to a certain sample
    frequency given by ``step``.

    Parameters
    ----------
    trajs : list[Trajectory]
        Input list of trajectories.
    step : int
        Numer of sample points.
    """

    def __init__(self, trajs: list[Trajectory], step: int = 1) -> None:
        check_same_dim(trajs)

        self.trajs = trajs
        self.step = step
        self.__speeds: np.ndarray | None = None

    @property
    def speeds(self) -> np.ndarray:
        """
        Get the speeds of the trajectories.

        Returns
        -------
        np.ndarray
            Concatenated array of speeds.
        """
        if self.__speeds is None:
            self.__speeds = np.concatenate(
                [subsample(traj, self.step).v.norm for traj in self.trajs]
            )
        return self.__speeds

    def plot(
        self,
        bins: int | None = None,
        show: bool = True,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot a histogram of the array of velocities ``v``.

        Parameters
        ----------
        v : np.ndarray
            Velocity array.
        show : bool, optional
            If True, the plot is shown. By default True.
        units : string, optional
            Velocity units. By default 'm/s'.

        Returns
        -------
        Axes
            Axes of the plot.
        """

        if "color" not in kwargs:
            kwargs["color"] = LIGHT_YELLOW

        if "density" in kwargs:
            kwargs.pop("density")

        plt.hist(self.speeds, bins=bins, ec=(0, 0, 0, 0.6), density=True, **kwargs)
        plt.xlabel(f"Speed [{self.trajs[0].units}]")
        plt.ylabel("pdf")
        plt.grid()
        ax = plt.gca() if ax is None else ax
        ax.set_axisbelow(True)

        if show:
            plt.show()

        return ax
