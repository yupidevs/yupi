from typing import List
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from yupi.trajectory import Trajectory


# TODO: Fix this implementation for dim != 2
def plot_trajectories(trajs: List[Trajectory], max_trajectories=None,
                      title=None, legend=True, show=True):
    """
    Plot all or ``max_trajectories`` trajectories from ```trajs``.

    Parameters
    ----------
    trajs : List[Trajectory]
        Input trajectories.
    max_trajectories : int, optional
        Number of trajectories to plot, by default None.
    title : str, optional
        Title of the plot, by default None.
    legend : bool, optional
        If True, legend is shown. By default True.
    show : bool, optional
        If Tue, the plot is shown. By default True.
    """

    if max_trajectories is None:
        max_trajectories = len(trajs)

    for i, t in enumerate(trajs):
        if i == max_trajectories:
            break

        # Plotting
        x, y = t.r.x, t.r.y
        traj_plot = plt.plot(x, y, '-')
        color = traj_plot[-1].get_color()
        plt.plot(x[0], y[0], 'o', mfc='white', zorder=2,
                 label=f'{t.id} initial position', color=color)
        plt.plot(x[-1], y[-1], 'o', mfc='white', zorder=2, color=color)
        plt.plot(x[-1], y[-1], 'o', alpha=.5,
                 label=f'{t.id} final position', color=color)

        if legend:
            plt.legend()

        plt.title(title)
        plt.tick_params(direction='in')
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')

    if show:
        plt.show()


def plot_trajectory(traj: Trajectory, title=None, legend=True, show=True):
    """
    Plot a single trajectory.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory.
    title : str, optional
        Title of the plot, by default None.
    legend : bool, optional
        If True, legend is shown. By default True.
    show : bool, optional
        If Tue, the plot is shown. By default True.
    """

    plot_trajectories([traj], title=title, legend=legend, show=show)


def plot_velocity_hist(v, bins=20, show=True):
    """[summary]

    Parameters
    ----------
    v : [type]
        [description]
    bins : int, optional
        [description], by default 20
    show : bool, optional
        [description], by default True
    """

    plt.hist(v, bins, density=True, ec='k', color='#fdd693')
    plt.xlabel('speed [m/s]')
    plt.ylabel('pdf')
    if show:
        plt.show()


def plot_angle_distribution(ang, bins=50, ax=None, show=True):
    """[summary]

    Parameters
    ----------
    ang : [type]
        [description]
    bins : int, optional
        [description], by default 50
    ax : [type], optional
        [description], by default None
    plot : bool, optional
        [description], by default True
    """

    if ax is None:
        ax = plt.gca(projection='polar')
    plt.hist(ang, bins, density=True, ec='k', color='.85')
    ax.set_theta_zero_location('N')
    ax.set_rlabel_position(135)
    ax.set_axisbelow(True)
    plt.xlabel('turning angles pdf')
    if show:
        plt.show()


def plot_msd(msd, msd_std, dt, lag=30, show=True):
    """[summary]

    Parameters
    ----------
    msd : [type]
        [description]
    msd_std : [type]
        [description]
    dt : [type]
        [description]
    lag : int, optional
        [description], by default 30
    show : bool, optional
        [description], by default True
    """

    lag_t_msd = dt * np.arange(lag)
    plt.plot(lag_t_msd, msd, color='.2')
    plt.fill_between(lag_t_msd, msd + msd_std, msd - msd_std, color='#afc0da')
    plt.xlabel('lag time [s]')
    plt.ylabel(r'$\mathrm{msd \; [m^2/s]}$')
    if show:
        plt.show()


def plot_kurtosis(kurtosis, dt=None, t_array=None, show=True):
    """[summary]

    Parameters
    ----------
    kurtosis : [type]
        [description]
    dt : [type], optional
        [description], by default None
    t_array : [type], optional
        [description], by default None
    plot : bool, optional
        [description], by default True
    """

    if dt:
        t_array = np.linspace(0, dt*len(kurtosis), len(kurtosis))
    if t_array is not None:
        plt.plot(t_array, kurtosis)
        plt.xlabel('time [s]')
    else:
        plt.plot(kurtosis)

    plt.ylabel('kurtosis')
    if show:
        plt.show()


def plot_vacf(vacf, dt, lag=50, show=True):
    """[summary]

    Parameters
    ----------
    vacf : [type]
        [description]
    dt : [type]
        [description]
    lag : int, optional
        [description], by default 50
    show : bool, optional
        [description], by default True
    """

    lag_t_vacf = dt * np.arange(lag)

    plt.plot(lag_t_vacf, vacf, '.', color='#870e11', mfc='w')
    plt.xlabel('lag time [s]')
    plt.ylabel(r'$\mathrm{vacf \; [(m/s)^2]}$')

    ax = plt.gca()

    inset_axes(ax, width='60%', height='60%', bbox_to_anchor=(0, 0, 1, 1),
               bbox_transform=ax.transAxes, loc='upper right')

    plt.plot(lag_t_vacf, vacf, '.', color='#870e11', mfc='w')
    plt.yscale('log')

    if show:
        plt.show()
