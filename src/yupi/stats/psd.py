from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from yupi._checkers import check_same_dt, check_uniform_time_spaced
from yupi.stats.vacf import VacfTimeAvgStat
from yupi.trajectory import Trajectory


class PsdStat:
    """
    Estimate the power spectral density of a list of Trajectory object
    as the Fourier transform of its velocity autocorrelation function.

    Parameters
    ----------
    trajs : list[Trajectory]
        Input list of trajectories.
    lag : int
        Number of steps that multiplied by ``dt`` defines the lag
        time.
    omega: bool
        If True, return the angular frequency instead of the frequency.
    """

    def __init__(self, trajs: list[Trajectory], lag: int, omega: bool = False) -> None:
        check_same_dt(trajs)
        check_uniform_time_spaced(trajs)
        self.trajs = trajs
        self.lag = lag
        self.omega = omega
        self.__psd: np.ndarray | None = None
        self.__frec: np.ndarray | None = None
        self.__psd_mean: np.ndarray | None = None
        self.__psd_std: np.ndarray | None = None

    def _compute_psd(self) -> np.ndarray:
        """
        Compute the power spectral density for each trajectory
        in the list of trajectories.
        """
        _vacf = VacfTimeAvgStat(self.trajs, self.lag).vacf
        _ft = np.fft.fft(_vacf, axis=0) * self.trajs[0].dt
        _ft = np.fft.fftshift(_ft)
        return np.abs(_ft)

    @property
    def psd(self) -> np.ndarray:
        """
        Array of power spectral density with shape ``(lag, N)``,
        where ``N`` is the number of trajectories.
        """
        if self.__psd is None:
            self.__psd = self._compute_psd()
        return self.__psd

    @property
    def psd_mean(self) -> np.ndarray:
        """Overall mean of the power spectral density"""
        if self.__psd_mean is None:
            self.__psd_mean = np.mean(self.psd, axis=1)
        return self.__psd_mean

    @property
    def psd_std(self) -> np.ndarray:
        """Overall standard deviation of the power spectral density"""
        if self.__psd_std is None:
            self.__psd_std = np.std(self.psd, axis=1)
        return self.__psd_std

    @property
    def frecuency(self) -> np.ndarray:
        """
        Frequency vector of the power spectral density.
        """
        if self.__frec is None:
            frec = 2 * np.pi * np.fft.fftfreq(self.lag, self.trajs[0].dt)
            frec = np.fft.fftshift(frec)
            self.__frec = frec * 2 * np.pi if self.omega else frec
        return self.__frec

    def plot(
        self,
        show: bool = True,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot the Power Spectral Density.

        Parameters
        ----------
        show : bool, optional
            If True, the plot is shown. By default True.

        Returns
        -------
        Axes
            Axes of the plot.
        """

        plt.plot(self.frecuency, self.psd_mean, label="psd", **kwargs)
        ax = plt.gca() if ax is None else ax
        if self.psd_std is not None:
            plt.fill_between(
                self.frecuency,
                self.psd_mean - self.psd_std,
                self.psd_mean + self.psd_std,
                alpha=0.3,
                label="psd_std",
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.grid()
        time_unit = self.trajs[0].units.time
        x_unit = f"rad/{time_unit}" if self.omega else "Hz"
        plt.xlabel(f"frequency [{x_unit}]")
        plt.ylabel("psd")
        plt.legend()
        if show:
            plt.show()

        return ax
