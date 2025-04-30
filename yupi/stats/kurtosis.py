from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from yupi._checkers import (
    check_same_dim,
    check_same_dt,
    check_same_t,
    check_uniform_time_spaced,
)
from yupi.graphics._style import GREEN, LIGHT_GREEN
from yupi.trajectory import Trajectory
from yupi.vector import Vector


def kurtosis_reference(trajs: list[Trajectory]) -> float:
    """Get the sampled kurtosis for the case of
    ``len(trajs)`` trajectories whose position
    vectors are normally distributed.

    Parameters
    ----------
    trajs : list[Trajectory]
        Input trajectories.

    Returns
    -------
    float
        Value of kurtosis.
    """
    check_same_dim(trajs)

    dim = trajs[0].dim
    count = len(trajs)
    kurt = dim * (dim + 2)
    if count == 1:
        return kurt
    return kurt * (count - 1) / (count + 1)


def _kurtosis(arr: np.ndarray) -> float:
    """
    Compute the kurtosis of the array, `arr`.

    If `arr` is not a one-dimensional array, it should
    be a horizontal collection of column vectors.

    Parameters
    ----------
    arr : np.adarray
        Data for which the kurtosis is calculated.

    Returns
    -------
    float
        Kurtosis of the data set.
    """

    arr = np.squeeze(arr)

    # ONE-DIMENSIONAL CASE
    if len(arr.shape) == 1:
        # Subtract the mean position at every time instant
        arr_zm = arr - arr.mean()

        # Second and fourth central moments averaging
        # over repetitions
        m_2 = np.mean(arr_zm**2)
        m_4 = np.mean(arr_zm**4)

        # Compute kurtosis for those cases in which the
        # second moment is different from zero
        if m_2 == 0:
            return 0
        kurt = m_4 / m_2**2
        return kurt

    # MULTIDIMENSIONAL CASE
    # arr should have shape (dim, trials)
    # (i.e., a horizontal sequence of column vectors)

    # Subtract the mean position
    arr_zm = arr - arr.mean(1)[:, None]

    try:
        # Inverse of the estimated covariance matrix
        cov_inv = np.linalg.inv(np.cov(arr))
    except np.linalg.LinAlgError:
        # Exception for the case of singular matrices
        return 0

    # Kurtosis definition for multivariate r.v.'s
    _k = np.sum(arr_zm * (cov_inv @ arr_zm), axis=0)
    kurt = np.mean(_k**2)

    return kurt


class KurtosisTimeAvgStat:
    """
    Estimate the kurtosis for every Trajectory object stored
    in ``trajs``.

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
        self.__kurtosis: np.ndarray | None = None
        self.__kurtosis_mean: np.ndarray | None = None
        self.__kurtosis_std: np.ndarray | None = None

    def _compute_kurtosis(self) -> np.ndarray:
        kurt = []
        for traj in self.trajs:
            current_kurt = np.empty(self.lag)
            for lag_ in range(self.lag):
                try:
                    lagged_r = traj.r[lag_:] - traj.r[:-lag_]
                except ValueError:
                    current_kurt[lag_] = 0
                    continue
                current_kurt[lag_] = _kurtosis(lagged_r.T)
            kurt.append(current_kurt)
        return np.transpose(kurt)

    @property
    def kurtosis(self) -> np.ndarray:
        """
        Array of kurtosis with shape ``(lag, N)``, where ``N`` is
        the number of trajectories.
        """
        if self.__kurtosis is None:
            self.__kurtosis = self._compute_kurtosis()
        return self.__kurtosis

    @property
    def kurtosis_mean(self) -> np.ndarray:
        """Overall mean of the kurtosis"""
        if self.__kurtosis_mean is None:
            self.__kurtosis_mean = np.mean(self.kurtosis, axis=1)
        return self.__kurtosis_mean

    @property
    def kurtosis_std(self) -> np.ndarray:
        """Overall standard deviation of the kurtosis"""
        if self.__kurtosis_std is None:
            self.__kurtosis_std = np.std(self.kurtosis, axis=1)
        return self.__kurtosis_std


class KurtosisStat:
    """
    Estimate kurtosis as a function of time of the
    list of Trajectory objects, ``trajs``. The average
    is perform over the ensemble of realizations.

    Parameters
    ----------
    trajs : list[Trajectory]
        Input list of trajectories.
    """

    def __init__(self, trajs: list[Trajectory]) -> None:
        check_same_dim(trajs)
        check_same_t(trajs)
        self.trajs = trajs
        self.__kurtosis: np.ndarray | None = None

    def _compute_kurtosis(self) -> np.ndarray:
        # Get ensemble positions where axis 0/1/2 are
        # in the order trials/time/dim
        r = Vector([traj.r for traj in self.trajs])

        # Set trials as the last axis
        moved_r = np.moveaxis(r, 0, 2)

        # Compute kurtosis at every time instant (loop over time)
        kurt = [_kurtosis(r_) for r_ in moved_r]

        return np.array(kurt)

    @property
    def kurtosis(self) -> np.ndarray:
        """Kurtosis at every time instant."""
        if self.__kurtosis is None:
            self.__kurtosis = self._compute_kurtosis()
        return self.__kurtosis

    def plot(
        self,
        show_ref: bool = True,
        ax: Axes | None = None,
        show: bool = True,
        ref_color: Any = LIGHT_GREEN,
        **kwargs: Any,
    ) -> Axes:
        """Plot kurtosis.

        Parameters
        ----------
        show_ref : bool, optional
            If True, the reference value is shown. By default True.
        ax : Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        show : bool, optional
            If True, the plot is shown. By default True.
        ref_color : Any, optional
            Color of the fill between the upper and lower bound.
            By default LIGHT_GREEN.

        Returns
        -------
        Axes
            Axes of the plot.
        """

        units = f" [{self.trajs[0].units.time}]"

        if "color" not in kwargs:
            kwargs["color"] = GREEN

        dt = self.trajs[0].dt
        t_array = np.linspace(0, dt * len(self.kurtosis), len(self.kurtosis))
        plt.plot(t_array, self.kurtosis, **kwargs)
        ax = plt.gca() if ax is None else ax

        if show_ref:
            kurtosis_ref = kurtosis_reference(self.trajs)
            bound_1 = self.kurtosis
            bound_2 = [kurtosis_ref] * len(t_array)
            plt.fill_between(t_array, bound_1, bound_2, color=ref_color)
        plt.xlabel(f"time{units}")

        plt.ylabel("kurtosis")
        plt.grid()
        if show:
            plt.show()

        return ax
