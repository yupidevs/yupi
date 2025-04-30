"""
This contains spatial plotting functions for the trajectories.
"""

import itertools
import logging
from typing import Any, Callable, Collection

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D

from yupi._checkers import check_exact_dim
from yupi.graphics._style import LINE, YUPI_COLORS
from yupi.trajectory import Trajectory


def _resolve_colors(total: int, color: Any) -> list[Any]:
    """
    Resolve the colors for a plot with multiple trajectories.

    Parameters
    ----------
    total : int
        Total number of trajectories.
    color : Any
        Color to use. If None, a cycle of colors is used.

    Returns
    -------
    list[Any]
        List of colors for each trajectory.
    """
    if color is None:
        cycle = itertools.cycle(YUPI_COLORS)
        return [next(cycle) for _ in range(total)]

    if isinstance(color, list):
        cycle = itertools.cycle(color)
        return [next(cycle) for _ in range(total)]

    return [color] * total


def _plot_2d_connections(trajs: list[Trajectory]) -> None:
    lengths = list(map(len, trajs))
    min_len = min(lengths)
    max_len = max(lengths)
    if min_len != max_len:
        logging.warning("Not all the trajectories have the same length.")
    for i in range(min_len):
        traj_points = [t[i] for t in trajs]
        traj_points.append(traj_points[0])
        for tp1, tp2 in itertools.pairwise(traj_points):
            seg_x = [tp1.r[0], tp2.r[0]]
            seg_y = [tp1.r[1], tp2.r[1]]
            plt.plot(seg_x, seg_y, color=(0.2, 0.2, 0.2), linewidth=0.5)


def plot_2d(
    trajs: list[Trajectory] | Trajectory,
    line_style: str = LINE,
    title: str | None = None,
    legend: bool = True,
    show: bool = True,
    connected: bool = False,
    color: Any = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """
    Plot all the points of trajectories from ``trajs`` in a 2D plane.

    Parameters
    ----------
    trajs : list[Trajectory] | Trajectory
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
    ax : matplotlib.axes.Axes, optional
        Axes where the plot is drawn, by default None.

        If None, then the current axes is used.
    """

    if isinstance(trajs, Trajectory):
        trajs = [trajs]

    check_exact_dim(trajs, 2)

    units = f" [{trajs[0].units.dist}]"

    if ax is None:
        ax = plt.gca()

    colors = _resolve_colors(len(trajs), color)

    if connected:
        _plot_2d_connections(trajs)

    for i, traj in enumerate(trajs):
        # Plotting
        x_data, y_data = traj.r.x, traj.r.y

        kwargs["color"] = colors[i]
        traj_plot = plt.plot(x_data, y_data, line_style, **kwargs)
        color = traj_plot[-1].get_color()
        traj_id = traj.traj_id if traj.traj_id else f"traj {i}"
        plt.plot(
            x_data[0],
            y_data[0],
            "o",
            mfc="white",
            zorder=2,
            label=f"{traj_id} start",
            color=color,
        )
        plt.plot(x_data[-1], y_data[-1], "o", mfc="white", zorder=2, color=color)
        plt.plot(
            x_data[-1],
            y_data[-1],
            "o",
            alpha=0.5,
            label=f"{traj_id} end",
            color=color,
        )

        if legend:
            plt.legend()

        if title is not None:
            plt.title(title)
        plt.tick_params(direction="in")
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel(f"x{units}")
        plt.ylabel(f"y{units}")

    if show:
        plt.show()

    return ax


def _plot_3d_connections(trajs: list[Trajectory], ax: Axes) -> None:
    lengths = list(map(len, trajs))
    min_len = min(lengths)
    max_len = max(lengths)
    if min_len != max_len:
        logging.warning("Not all the trajectories have the same length.")
    for i in range(min_len):
        traj_points = [t[i] for t in trajs]
        traj_points.append(traj_points[0])
        for tp1, tp2 in itertools.pairwise(traj_points):
            seg_x = [tp1.r[0], tp2.r[0]]
            seg_y = [tp1.r[1], tp2.r[1]]
            seg_z = [tp1.r[2], tp2.r[2]]
            ax.plot(seg_x, seg_y, seg_z, color=(0.2, 0.2, 0.2), linewidth=0.5)


def plot_3d(
    trajs: list[Trajectory] | Trajectory,
    line_style: str = LINE,
    title: str | None = None,
    legend: bool = True,
    show: bool = True,
    connected: bool = False,
    color: Any = None,
    ax: Axes3D | None = None,
    **kwargs: Any,
) -> Axes3D:
    """
    Plot all the points of trajectories from ``trajs`` in a 3D space.

    Parameters
    ----------
    trajs : list[Trajectory] | Trajectory
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
    ax : matplotlib.axes.Axes, optional
        Axes where the plot is drawn, by default None.

        If None, then a new axes is created with projection='3d'.
    """

    if isinstance(trajs, Trajectory):
        trajs = [trajs]

    check_exact_dim(trajs, 3)

    units = f" [{trajs[0].units.dist}]"

    colors = _resolve_colors(len(trajs), color)

    ax = plt.axes(projection="3d") if ax is None else ax

    assert isinstance(ax, Axes3D), "ax must be a 3D Axes"

    if connected:
        _plot_3d_connections(trajs, ax)

    for i, traj in enumerate(trajs):
        # Plotting
        x_data, y_data, z_data = traj.r.x, traj.r.y, traj.r.z

        kwargs["color"] = colors[i]
        traj_plot = ax.plot(x_data, y_data, z_data, line_style, **kwargs)
        color = traj_plot[-1].get_color()
        traj_id = traj.traj_id if traj.traj_id else f"traj {i}"

        ax.plot(
            x_data[0],
            y_data[0],
            z_data[0],
            "o",
            mfc="white",
            label=f"{traj_id} start",
            color=color,
        )

        ax.plot(x_data[-1], y_data[-1], z_data[-1], "o", mfc="white", color=color)
        ax.plot(
            x_data[-1],
            y_data[-1],
            z_data[-1],
            "o",
            alpha=0.5,
            label=f"{traj_id} end",
            color=color,
        )

        if legend:
            plt.legend()

        if title is not None:
            plt.title(title)
        plt.tick_params(direction="in")
        plt.grid(True)
        ax.set_xlabel(f"x{units}")
        ax.set_ylabel(f"y{units}")
        ax.set_zlabel(f"z{units}")

    if show:
        plt.show()

    return ax


def plot_vs_time(
    trajs: list[Trajectory] | Trajectory,
    key: Callable[[Trajectory], Collection[float]],
    line_style: str = LINE,
    y_label: str | None = None,
    title: str | None = None,
    legend: bool = True,
    color: Any = None,
    show: bool = True,
    **kwargs: Any,
) -> Axes:
    if isinstance(trajs, Trajectory):
        trajs = [trajs]

    x_units = f"time [{trajs[0].units.time}]"
    colors = _resolve_colors(len(trajs), color)

    for i, traj in enumerate(trajs):
        kwargs["color"] = colors[i]
        y_data = np.array(key(traj))
        x_data = traj.t
        traj_id = traj.traj_id if traj.traj_id else f"traj {i}"
        plt.plot(x_data, y_data, line_style, **kwargs, label=traj_id)
        plt.xlabel(x_units)
        if y_label is not None:
            plt.ylabel(y_label)
        plt.grid()

        if title is not None:
            plt.title(title)

    if legend:
        plt.legend()

    if show:
        plt.show()

    return plt.gca()
