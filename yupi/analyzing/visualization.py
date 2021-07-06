import logging
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from yupi.trajectory import Trajectory


LINE = '-'
DOTTED = 'o'
LINE_DOTTED = '-o'

# TODO: Fix this implementation for dim != 2
def plot_trajectories(trajs: List[Trajectory], line_style: str = LINE,
                      title: str = None, legend: bool = True,
                      show: bool = True, connected: bool = False,
                      unit: str = 'm', color = None, **kwargs):
    """
    Plot trajectories from ```trajs``.

    Parameters
    ----------
    trajs : List[Trajectory]
        Input trajectories.
    line_style : str
        Type of the trajectory line to plot. It uses the matplotlib,
        notation, by default '-'.
    title : str, optional
        Title of the plot, by default None.
    legend : bool, optional
        If True, legend is shown. By default True.
    show : bool, optional
        If True, the plot is shown. By default True.
    connected : bool
        If True, all the trajectory points of same index will be,
        connected.

        If the trajectories do not have same length then the points
        will be connected until the shortest trajectory last index.
    color : str or tuple or list
        Defines the color of the trajectories, by default None.

        If color is of type ``str`` or ``tuple`` (rgb) then the color
        is applied to all trajectories. If color is of type ``list``
        then the trajectories take the color according to the index.

        If there are less colors than trajectories then the remaining
        trajectories are colored automatically (not with the same
        color).
    """

    unit = '' if unit is None else f' [{unit}]'

    colors = None
    if color is not None:
        if isinstance(color, (str, tuple)):
            kwargs['color'] = color
        elif isinstance(color, list):
            colors = color

    if connected:
        lengths = list(map(len, trajs))
        min_len = min(lengths)
        max_len = max(lengths)
        if min_len != max_len:
            logging.warning('Not all the trajectories have the same length.')
        for i in range(min_len):
            traj_points = [t[i] for t in trajs]
            traj_points.append(traj_points[0])
            for tp1, tp2 in zip(traj_points[:-1], traj_points[1:]):
                xs = [tp1.r[0], tp2.r[0]]
                ys = [tp1.r[1], tp2.r[1]]
                plt.plot(xs, ys, color=(0.2, 0.2, 0.2), linewidth=0.5)

    for i, t in enumerate(trajs):

        # Plotting
        x, y = t.r.x, t.r.y
        if colors is not None:
            if i < len(colors):
                kwargs['color'] = colors[i]
            else:
                kwargs.pop('color')
        traj_plot = plt.plot(x, y, line_style, **kwargs)
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
        plt.xlabel(f'x{unit}')
        plt.ylabel(f'y{unit}')

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
