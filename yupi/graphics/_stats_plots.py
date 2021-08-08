import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from yupi.graphics._style import (
    LIGHT_YELLOW,
    LIGHT_BLUE,
    LIGHT_ORANGE,
    GREEN,
    LIGHT_GREEN,
    RED
)

def _validate_units(units):
    return '' if units is None else f' [{units}]'

def plot_velocity_hist(v, show: bool = True, units: str = 'm/s', **kwargs):
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

    if 'color' not in kwargs:
        kwargs['color'] = LIGHT_YELLOW

    if 'density' in kwargs:
        kwargs.pop('density')

    units = _validate_units(units)

    plt.hist(v, ec=(0,0,0,0.6), density=True, **kwargs)
    plt.xlabel(f'speed{units}')
    plt.ylabel('pdf')
    plt.grid()
    plt.gca().set_axisbelow(True)

    if show:
        plt.show()


def plot_angles_hist(ang, bins, show: bool = True, ax=None):
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
        ax = plt.gca(projection='polar')
    plt.hist(ang, bins, density=True, ec=(0,0,0,0.6), color=LIGHT_BLUE)
    ax.set_theta_zero_location('N')
    ax.set_rlabel_position(135)
    ax.set_axisbelow(True)
    plt.xlabel('turning angles pdf')
    if show:
        plt.show()


def plot_msd(msd, msd_std, dt, lag, x_units: str = 's', y_units: str = 'm^2/s',
             show=True):
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
    plt.plot(lag_t_msd, msd, color='.2')
    upper_bound = msd + msd_std
    lower_bound = msd - msd_std
    plt.fill_between(lag_t_msd, upper_bound, lower_bound, color=LIGHT_ORANGE)
    plt.xlabel(f'lag time{x_units}')
    plt.ylabel(r'$\mathrm{msd \;' + y_units + '}$')
    plt.grid()
    if show:
        plt.show()


def plot_kurtosis(kurtosis, dt=None, t_array=None, kurtosis_ref: float = None,
                  units: str = 's', show=True):
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

    if dt:
        t_array = np.linspace(0, dt*len(kurtosis), len(kurtosis))
    if t_array is not None:
        plt.plot(t_array, kurtosis, color=GREEN)

        if kurtosis_ref is not None:
            bound_1 = kurtosis
            bound_2 = [kurtosis_ref]*len(t_array)
            plt.fill_between(t_array, bound_1, bound_2, color=LIGHT_GREEN)
        plt.xlabel(f'time{units}')
    else:
        plt.plot(kurtosis)

    plt.ylabel('kurtosis')
    plt.grid()
    if show:
        plt.show()


def plot_vacf(vacf, dt, lag, x_units: str = 's', y_units: str = '(m/s)^2',
              show=True):
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
    show : bool, optional
        If True, the plot is shown. By default True.
    """

    x_units = _validate_units(x_units)
    y_units = _validate_units(y_units)

    lag_t_vacf = dt * np.arange(lag)

    plt.plot(lag_t_vacf, vacf, '.', color=RED)
    plt.xlabel(f'lag time{x_units}')
    plt.ylabel(r'$\mathrm{vacf \;' + y_units + '}$')
    plt.grid()

    ax = plt.gca()

    inset_axes(ax, width='60%', height='60%', bbox_to_anchor=(0, 0, 1, 1),
               bbox_transform=ax.transAxes, loc='upper right')

    plt.plot(lag_t_vacf, vacf, '.', color=RED)
    plt.yscale('log')
    plt.grid()

    if show:
        plt.show()


def plot_psd(psd_mean, omega, psd_std=None, show=True):
    """Plot the Power Spectral Density.

    Parameters
    ----------
    psd_mean : np.ndarray
        Power spectral density array.
    omega : np.ndarray
        Array of angular frequencies.
    psd_std : np.ndarray, optional
        Standard deviation of the power spectrum.
        By default None.
    show : bool, optional
        If True, the plot is shown. By default True.
    """

    plt.plot(omega, psd_mean, label='psd')
    if psd_std is not None:
        plt.fill_between(omega, psd_mean - psd_std, psd_mean + psd_std,
            alpha=.3, label='psd_std')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    if show:
        plt.show()
