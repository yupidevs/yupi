from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from yupi._checkers import (
    check_same_dim,
    check_same_dt,
    check_same_length,
    check_same_t,
    check_uniform_time_spaced,
)
from yupi.graphics._style import LIGHT_ORANGE
from yupi.trajectory import Trajectory


class MsdTimeAvgStat:
    """
    Estimate the mean square displacement for every Trajectory
    object stored in ``trajs`` as the average of the square of
    dispacement vectors as a function of the lag time.

    This is a convenience estimator specially when trajectories
    do not have equal lengths.

    Parameters
    ----------
    trajs : list[Trajectory]
        Input list of trajectories.
    lag : int
        Number of steps that multiplied by ``dt`` defines the lag
        time.
    """

    def __init__(self, trajs: list[Trajectory], lag: int):
        check_same_dim(trajs)
        check_same_dt(trajs)
        check_uniform_time_spaced(trajs)
        self.trajs = trajs
        self.lag = lag
        self.__msd: np.ndarray | None = None
        self.__msd_mean: np.ndarray | None = None
        self.__msd_std: np.ndarray | None = None

    @property
    def msd(self) -> np.ndarray:
        """
        Calculate the mean square displacement for each trajectory.

        Returns
        -------
        np.ndarray
            Array of mean square displacements with shape ``(lag, N)``,
            where ``N`` the number of trajectories.
        """
        if self.__msd is None:
            _msd = []
            for traj in self.trajs:
                # Position vectors
                r = traj.r

                # Compute msd for a single trajectory
                current_msd = np.empty(self.lag)
                for lag_ in range(1, self.lag + 1):
                    # Lag displacement vectors
                    lagged_r = r[lag_:] - r[:-lag_]
                    # Lag displacement
                    dr2 = np.sum(lagged_r**2, axis=1)
                    # Averaging over a single realization
                    current_msd[lag_ - 1] = np.mean(dr2)

                # Append all square displacements
                _msd.append(current_msd)

            # Transpose to have time/trials as first/second axis
            self.__msd = np.transpose(_msd)

        return self.__msd

    @property
    def msd_mean(self) -> np.ndarray:
        """Overall mean square displacement."""
        if self.__msd_mean is None:
            self.__msd_mean = np.mean(self.msd, axis=1)
        return self.__msd_mean

    @property
    def msd_std(self) -> np.ndarray:
        """Overall standard deviation of the mean square displacement."""
        if self.__msd_std is None:
            self.__msd_std = np.std(self.msd, axis=1)
        return self.__msd_std

    def plot(
        self,
        fill_color: Any = LIGHT_ORANGE,
        ax: Axes | None = None,
        show: bool = True,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot the mean square displacement.

        Parameters
        ----------
        fill_color : Any, optional
            Color of the fill between the mean and standard deviation.
            Default is LIGHT_ORANGE.
        ax : Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        show : bool, optional
            Whether to show the plot or not. Default is True.
        """

        dt = self.trajs[0].dt
        units = self.trajs[0].units
        lag_t_msd = dt * np.arange(self.lag)
        default_kwargs: Any = {"color": ".2"}
        default_kwargs.update(kwargs)
        plt.plot(lag_t_msd, self.msd_mean, ".", **default_kwargs)
        upper_bound = self.msd_mean + self.msd_std
        lower_bound = self.msd_mean - self.msd_std
        ax = plt.gca() if ax is None else ax
        plt.fill_between(lag_t_msd, upper_bound, lower_bound, color=fill_color)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(f"Lag time [{units.time}]")
        plt.ylabel(r"$\mathrm{msd \;" + str(units.dist) + "^2}$")
        plt.grid()
        if show:
            plt.show()

        return ax


class MsdStat:
    """
    Compute the square displacements for every Trajectory object
    stored in ``trajs`` as the square of the current position vector
    that has been subtracted the initial position.

    Trajectories should have the same length.

    Parameters
    ----------
    trajs : list[Trajectory]
        Input list of trajectories.
    """

    def __init__(self, trajs: list[Trajectory]):
        check_same_length(trajs)
        check_same_dim(trajs)
        check_same_t(trajs)
        self.trajs = trajs
        self.__msd: np.ndarray | None = None
        self.__msd_mean: np.ndarray | None = None
        self.__msd_std: np.ndarray | None = None

    @property
    def msd(self) -> np.ndarray:
        """
        Calculate the mean square displacement for each trajectory.

        Returns
        -------
        np.ndarray
            Array of mean square displacements with shape ``(lag, N)``,
            where ``N`` the number of trajectories.
        """
        if self.__msd is None:
            _msd = []
            for traj in self.trajs:
                # Position vectors
                r = traj.r

                # Square displacements
                r_2 = (r - r[0]) ** 2  # Square coordinates
                r_2_dis = np.sum(r_2, axis=1)  # Square distances
                _msd.append(r_2_dis)  # Append square distances

            # Transpose to have time/trials as first/second axis
            self.__msd = np.transpose(_msd)

        return self.__msd

    @property
    def msd_mean(self) -> np.ndarray:
        """Overall mean square displacement."""
        if self.__msd_mean is None:
            self.__msd_mean = np.mean(self.msd, axis=1)
        return self.__msd_mean

    @property
    def msd_std(self) -> np.ndarray:
        """Overall standard deviation of the mean square displacement."""
        if self.__msd_std is None:
            self.__msd_std = np.std(self.msd, axis=1)
        return self.__msd_std
