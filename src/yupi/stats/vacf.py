from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from yupi._checkers import (
    check_same_dim,
    check_same_dt,
    check_same_t,
    check_uniform_time_spaced,
)
from yupi.graphics._style import RED
from yupi.trajectory import Trajectory


class VacfTimeAvgStat:
    """
    Estimate the velocity autocorrelation function for every
    Trajectory object stored in ``trajs`` as the average of the
    dot product between velocity vectors that are distant a certain
    lag time.

    This is a convenience estimator specially when trajectories do
    not have equal lengths.

    Parameters
    ----------
    trajs : list[Trajectory]
        Input list of trajectories.
    lag : int
        Number of steps that multiplied by ``dt`` defines the lag
        time.
    """

    def __init__(self, trajs: list[Trajectory], lag: int) -> None:
        check_same_dim(trajs)
        check_same_dt(trajs)
        check_uniform_time_spaced(trajs)
        self.trajs = trajs
        self.lag = lag
        self.__vacf: np.ndarray | None = None
        self.__vacf_mean: np.ndarray | None = None
        self.__vacf_std: np.ndarray | None = None

    def _compute_vacf(self) -> np.ndarray:
        _vacf = []
        for traj in self.trajs:
            # Cartesian velocity components
            v = traj.v

            # Compute vacf for a single trajectory
            current_vacf = np.empty(self.lag)
            for lag_ in range(1, self.lag + 1):
                # Multiply components given lag
                v1, v2 = v[:-lag_], v[lag_:]
                v1v2 = (v1 - v1.mean(axis=0)) * (v2 - v2.mean(axis=0))

                # Dot product for a given lag time
                v1_dot_v2 = np.sum(v1v2, axis=1)

                # Averaging over a single realization
                current_vacf[lag_ - 1] = np.mean(v1_dot_v2)

            # Append the vacf for a every single realization
            _vacf.append(current_vacf)

        # Aranspose to have time/trials as first/second axis
        return np.transpose(_vacf)

    @property
    def vacf(self) -> np.ndarray:
        """
        Array of velocity autocorrelation function with shape
        ``(lag, N)``, where ``N`` is the number of trajectories.
        """
        if self.__vacf is None:
            self.__vacf = self._compute_vacf()
        return self.__vacf

    @property
    def vacf_mean(self) -> np.ndarray:
        """Overall mean of the velocity autocorrelation function"""
        if self.__vacf_mean is None:
            self.__vacf_mean = np.mean(self.vacf, axis=1)
        return self.__vacf_mean

    @property
    def vacf_std(self) -> np.ndarray:
        """Overall standard deviation of the velocity autocorrelation function"""
        if self.__vacf_std is None:
            self.__vacf_std = np.std(self.vacf, axis=1)
        return self.__vacf_std

    def plot(
        self,
        log_inset: bool = True,
        ax: Axes | None = None,
        show: bool = True,
        **kwargs: Any,
    ) -> Axes:
        """Plot Velocity Autocorrelation Function.

        Parameters
        ----------
        log_inset : bool, optional
            If True, a log-log inset is shown. By default True.
        ax : Axes, optional
            Axes to plot on. By default None.
        show : bool, optional
            If True, the plot is shown. By default True.

        Returns
        -------
        Axes
            Axes of the plot.
        """

        x_units = f" [{self.trajs[0].units.time}]"
        y_units = f" [({self.trajs[0].units})^2]"

        lag_t_vacf = self.trajs[0].dt * np.arange(self.lag)

        if "color" not in kwargs:
            kwargs["color"] = RED

        plt.plot(lag_t_vacf, self.vacf_mean, ".", **kwargs)
        plt.xlabel(f"Lag time {x_units}")
        plt.ylabel(r"$\mathrm{vacf \;" + y_units + "}$")
        plt.grid()

        ax = plt.gca() if ax is None else ax

        if log_inset:
            inset_axes(
                ax,
                width="60%",
                height="60%",
                bbox_to_anchor=(0, 0, 1, 1),
                bbox_transform=ax.transAxes,
                loc="upper right",
            )

            plt.plot(lag_t_vacf, self.vacf_mean, ".", **kwargs)
            plt.yscale("log")
            plt.grid()

        if show:
            plt.show()

        return ax


class VacfStat:
    """
    Compute the pair-wise dot product between initial and current
    velocity vectors for every Trajectory object stored in ``trajs``.

    Parameters
    ----------
    trajs : list[Trajectory]
        Input list of trajectories.
    """

    def __init__(self, trajs: list[Trajectory]) -> None:
        check_same_dim(trajs)
        check_same_t(trajs)
        self.trajs = trajs
        self.__vacf: np.ndarray | None = None
        self.__vacf_mean: np.ndarray | None = None
        self.__vacf_std: np.ndarray | None = None

    def _compute_vacf(self) -> np.ndarray:
        _vacf = []
        for traj in self.trajs:
            # Cartesian velocity components
            v = traj.v

            # Pair-wise dot product between velocities at t0 and t
            v0_dot_v = np.sum(v[0] * v, axis=1)

            # Append all veloctiy dot products
            _vacf.append(v0_dot_v)

        # Transpose to have time/trials as first/second axis
        return np.transpose(_vacf)

    @property
    def vacf(self) -> np.ndarray:
        """
        Array of velocity dot products with shape ``(n, N)``, where
        ``n`` is the total number of time steps and ``N`` the number
        of trajectories.
        """
        if self.__vacf is None:
            self.__vacf = self._compute_vacf()
        return self.__vacf

    @property
    def vacf_mean(self) -> np.ndarray:
        """Overall mean of the velocity autocorrelation function"""
        if self.__vacf_mean is None:
            self.__vacf_mean = np.mean(self.vacf, axis=1)
        return self.__vacf_mean

    @property
    def vacf_std(self) -> np.ndarray:
        """Overall standard deviation of the velocity autocorrelation function"""
        if self.__vacf_std is None:
            self.__vacf_std = np.std(self.vacf, axis=1)
        return self.__vacf_std
