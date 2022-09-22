"""
This contains the plotting functions of the statistical observables.
"""

import itertools
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from yupi.graphics._style import (
    GREEN,
    LIGHT_BLUE,
    LIGHT_GREEN,
    LIGHT_ORANGE,
    LIGHT_YELLOW,
    RED,
    YUPI_COLORS,
    YUPI_LIGHT_COLORS,
    _plot_basic_properties,
)


def _validate_units(units):
    return "" if units is None else f" [{units}]"


@_plot_basic_properties
def plot_hist(values: np.ndarray, **kwargs):
    """
    Plot a histogram of the given

    Parameters
    ----------
    values : np.ndarray
        Input values
    """

    if "color" not in kwargs:
        kwargs["color"] = YUPI_LIGHT_COLORS[0]
    if "ec" not in kwargs:
        kwargs["ec"] = (0, 0, 0, 0.6)
    plt.hist(values, **kwargs)


@_plot_basic_properties
def plot_hists(
    values_list: List[np.ndarray],
    kwargs_list: Optional[List[dict]] = None,
    labels: Optional[List[str]] = None,
    filled: bool = False,
    **general_kwargs,
):
    """
    Plot several histograms given a collection of values.

    Parameters
    ----------
    values_list : List[np.ndarray]
        Collection of values.
    kwargs_list : Optional[List[dict]]
        kwargs of each plot, by default []

        If given, the length must be the same as the length of
        ``values``.
    """

    if kwargs_list and len(kwargs_list) != len(values_list):
        raise ValueError(
            "The length of 'kwargs_list' must be equals the length of 'values_list'"
        )

    kwargs_list = [{}] * len(values_list)

    cycle = itertools.cycle(YUPI_COLORS)
    colors = [next(cycle) for _ in values_list]
    ax = plt.gca()

    for i, vals in enumerate(values_list):
        kwargs = kwargs_list[i]
        kwargs = kwargs if len(kwargs) != 0 else general_kwargs
        color = kwargs.get("color", colors[i])
        alpha = kwargs.get("alpha", 0.3 if filled else 1)
        lw = kwargs.get("lw", 1.5)  # pylint: disable=invalid-name
        if labels is not None:
            kwargs["label"] = labels[i]
        if "histtype" in kwargs:
            kwargs.pop("histtype")

        plt.hist(vals, histtype="step", color=color, lw=lw, **kwargs)

        if filled:
            kwargs.pop("label", None)
            plt.hist(vals, histtype="stepfilled", color=color, alpha=alpha, **kwargs)

    return ax


def plot_speed_hist(v, show: bool = True, units: str = "m/s", **kwargs):
    """Plot a histogram of the array of velocities ``v``.

    Parameters
    ----------
    v : np.ndarray
        Velocity array.
    show : bool, optional
        If True, the plot is shown. By default True.
    units : string, optional
        Velocity units. By default 'm/s'.
    """

    if "color" not in kwargs:
        kwargs["color"] = LIGHT_YELLOW

    if "density" in kwargs:
        kwargs.pop("density")

    units = _validate_units(units)

    plt.hist(v, ec=(0, 0, 0, 0.6), density=True, **kwargs)
    plt.xlabel(f"speed{units}")
    plt.ylabel("pdf")
    plt.grid()
    ax = plt.gca()
    ax.set_axisbelow(True)

    if show:
        plt.show()

    return ax


def plot_angles_hist(ang, bins, show: bool = True, ax=None, **kwargs):
    """Plot a histogram of the array of angles ``ang``.

    Parameters
    ----------
    ang : np.ndarray
        Array of angles.
    bins: int
        Number of histogram bins.
    show : bool, optional
        If True, the plot is shown. By default True.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to plot. By default None.
    """

    if ax is None:
        ax = plt.axes(projection="polar")
    elif ax.name != "polar":
        raise ValueError("The axes must be polar")
    default_kwargs = {"color": LIGHT_BLUE, "ec": (0, 0, 0, 0.6), "density": True}
    default_kwargs.update(kwargs)
    plt.hist(ang, bins, **default_kwargs)
    ax.set_theta_zero_location("N")
    ax.set_rlabel_position(135)
    ax.set_axisbelow(True)
    plt.xlabel("turning angles pdf")
    if show:
        plt.show()

    return ax


def plot_msd(
    msd,
    msd_std,
    dt,
    lag,
    x_units: str = "s",
    y_units: str = "m^2",
    show=True,
    fill_color=LIGHT_ORANGE,
    **kwargs,
):
    """Plot Mean Square Displacement.

    Parameters
    ----------
    msd : np.ndarray
        Mean square displacement array.
    msd_std : np.ndarray
        Standard deviation.
    dt : float
        Trajectories time step.
    lag : int, optional
        Lag time.
    x_units : str, optional
        Units of the time axes.
    y_units : str, optional
        Units of the MSD axes.
    show : bool, optional
        If True, the plot is shown. By default True.
    """

    x_units = _validate_units(x_units)
    y_units = _validate_units(y_units)

    lag_t_msd = dt * np.arange(lag)
    default_kwargs = {"color": ".2"}
    default_kwargs.update(kwargs)
    plt.plot(lag_t_msd, msd, ".", **default_kwargs)
    upper_bound = msd + msd_std
    lower_bound = msd - msd_std
    ax = plt.gca()
    plt.fill_between(lag_t_msd, upper_bound, lower_bound, color=fill_color)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(f"lag time{x_units}")
    plt.ylabel(r"$\mathrm{msd \;" + y_units + "}$")
    plt.grid()
    if show:
        plt.show()

    return ax


def plot_kurtosis(
    kurtosis,
    dt=None,
    t_array=None,
    kurtosis_ref: Optional[float] = None,
    units: str = "s",
    show=True,
    ref_color=LIGHT_GREEN,
    **kwargs,
):
    """Plot kurtosis.

    Parameters
    ----------
    kurtosis : np.adarray
        Kurtosis array.
    dt : float
        Trajectories time step.
    t_array : np.ndarray, optional
        Array of time instants that match with every value in
        ``kurtosis``. By default None.
    kurtosis_ref : float, optional
        The value of kurtosis for a gaussian.
    units : str, optional
        Units of the time axes.
    show : bool, optional
        If True, the plot is shown. By default True.
    """

    units = _validate_units(units)

    if "color" not in kwargs:
        kwargs["color"] = GREEN

    if dt:
        t_array = np.linspace(0, dt * len(kurtosis), len(kurtosis))
    if t_array is not None:
        plt.plot(t_array, kurtosis, **kwargs)
        ax = plt.gca()

        if kurtosis_ref is not None:
            bound_1 = kurtosis
            bound_2 = [kurtosis_ref] * len(t_array)
            plt.fill_between(t_array, bound_1, bound_2, color=ref_color)
        plt.xlabel(f"time{units}")
    else:
        plt.plot(kurtosis, **kwargs)
        ax = plt.gca()

    plt.ylabel("kurtosis")
    plt.grid()
    if show:
        plt.show()

    return ax


def plot_vacf(
    vacf,
    dt,
    lag,
    x_units: str = "s",
    y_units: Optional[str] = "(m/s)^2",
    log_inset: bool = True,
    show: bool = True,
    **kwargs,
):
    """Plot Velocity Autocorrelation Function.

    Parameters
    ----------
    vacf : np.ndarray
        Velocity autocorrelation function array.
    dt : float
        Trajectories time step.
    lag : int
        Number of steps that multiplied by ``dt`` defines the lag
        time.
    x_units : str, optional
        Units of the time axes.
    y_units : str, optional
        Units of the VACF axes.
    log_inset : bool, optional
        If True, a log-log inset is shown. By default True.
    show : bool, optional
        If True, the plot is shown. By default True.
    """

    x_units = _validate_units(x_units)
    y_units = _validate_units(y_units)

    lag_t_vacf = dt * np.arange(lag)

    if "color" not in kwargs:
        kwargs["color"] = RED

    plt.plot(lag_t_vacf, vacf, ".", **kwargs)
    plt.xlabel(f"lag time{x_units}")
    plt.ylabel(r"$\mathrm{vacf \;" + y_units + "}$")
    plt.grid()

    ax = plt.gca()

    if log_inset:
        inset_axes(
            ax,
            width="60%",
            height="60%",
            bbox_to_anchor=(0, 0, 1, 1),
            bbox_transform=ax.transAxes,
            loc="upper right",
        )

        plt.plot(lag_t_vacf, vacf, ".", **kwargs)
        plt.yscale("log")
        plt.grid()

    if show:
        plt.show()

    return ax


def plot_psd(psd_mean, frec, psd_std=None, omega=True, show=True, **kwargs):
    """Plot the Power Spectral Density.

    Parameters
    ----------
    psd_mean : np.ndarray
        Power spectral density array.
    frec : np.ndarray
        Array of frequencies.
    psd_std : np.ndarray, optional
        Standard deviation of the power spectrum.
        By default None.
    omega : bool, optional
        If True, the `freq` is instended to be in rad/s, otherwise
        in Hz. By default True.
    show : bool, optional
        If True, the plot is shown. By default True.
    """

    plt.plot(frec, psd_mean, label="psd", **kwargs)
    ax = plt.gca()
    if psd_std is not None:
        plt.fill_between(
            frec, psd_mean - psd_std, psd_mean + psd_std, alpha=0.3, label="psd_std"
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    x_unit = "rad/s" if omega else "Hz"
    plt.xlabel(f"frequency [{x_unit}]")
    plt.ylabel("psd")
    plt.legend()
    if show:
        plt.show()

    return ax
