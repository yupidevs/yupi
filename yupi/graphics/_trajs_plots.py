"""
This contains spatial plotting functions for the trajectories.
"""

import itertools
import logging
import warnings
from typing import Callable, Collection, List, Optional, Union

import matplotlib.pyplot as plt

from yupi import Trajectory
from yupi.graphics._style import LINE, YUPI_COLORS


def plot_2d(
    trajs: Union[List[Trajectory], Trajectory],
    line_style: str = LINE,
    title: Optional[str] = None,
    legend: bool = True,
    show: bool = True,
    connected: bool = False,
    units: str = "m",
    color=None,
    **kwargs,
):
    """
    Plot all the points of trajectories from ``trajs`` in a 2D plane.

    Parameters
    ----------
    trajs : Union[List[Trajectory], Trajectory]
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

    if isinstance(trajs, Trajectory):
        trajs = [trajs]

    units = "" if units is None else f" [{units}]"
    ax = plt.gca()

    colors = itertools.cycle(YUPI_COLORS)
    if color is not None:
        if isinstance(color, (str, tuple)):
            colors = itertools.cycle([color])
        elif isinstance(color, list):
            colors = itertools.cycle(color)

    if connected:
        lengths = list(map(len, trajs))
        min_len = min(lengths)
        max_len = max(lengths)
        if min_len != max_len:
            logging.warning("Not all the trajectories have the same length.")
        for i in range(min_len):
            traj_points = [t[i] for t in trajs]
            traj_points.append(traj_points[0])
            for tp1, tp2 in zip(traj_points[:-1], traj_points[1:]):
                seg_x = [tp1.r[0], tp2.r[0]]
                seg_y = [tp1.r[1], tp2.r[1]]
                plt.plot(seg_x, seg_y, color=(0.2, 0.2, 0.2), linewidth=0.5)

    for i, traj in enumerate(trajs):

        if traj.dim != 2:
            logging.warning(
                "Using plot_2d with a trajectory of %i dimensions"
                " Trajectory No. %i with id %s",
                traj.dim,
                i,
                traj.traj_id,
            )

        # Plotting
        x_data, y_data = traj.r.x, traj.r.y

        kwargs["color"] = next(colors)
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

        plt.title(title)
        plt.tick_params(direction="in")
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel(f"x{units}")
        plt.ylabel(f"y{units}")

    if show:
        plt.show()

    return ax


def plot_2D(  # pylint: disable=invalid-name
    trajs: Union[List[Trajectory], Trajectory],
    line_style: str = LINE,
    title: Optional[str] = None,
    legend: bool = True,
    show: bool = True,
    connected: bool = False,
    units: str = "m",
    color=None,
    **kwargs,
):
    """
    .. deprecated:: 0.10.0
        :func:`plot_2D` will be removed in a future version, use
        :func:`plot_2d` instead.

    Plot all the points of trajectories from ``trajs`` in a 2D plane.

    Parameters
    ----------
    trajs : Union[List[Trajectory], Trajectory]
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

    warnings.warn(
        "plot_2D is deprecated and will be removed in a future version, "
        "use plot_2d instead",
        DeprecationWarning,
    )
    plot_2d(
        trajs,
        line_style,
        title,
        legend,
        show,
        connected,
        units,
        color,
        **kwargs,
    )


def plot_3d(
    trajs: Union[List[Trajectory], Trajectory],
    line_style: str = LINE,
    title: Optional[str] = None,
    legend: bool = True,
    show: bool = True,
    connected: bool = False,
    units: str = "m",
    color=None,
    **kwargs,
):
    """
    Plot all the points of trajectories from ``trajs`` in a 3D space.

    Parameters
    ----------
    trajs : Union[List[Trajectory], Trajectory]
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

    if isinstance(trajs, Trajectory):
        trajs = [trajs]

    units = "" if units is None else f" [{units}]"

    colors = itertools.cycle(YUPI_COLORS)
    if color is not None:
        if isinstance(color, (str, tuple)):
            colors = itertools.cycle([color])
        elif isinstance(color, list):
            colors = itertools.cycle(color)

    ax = plt.axes(projection="3d")

    if connected:
        lengths = list(map(len, trajs))
        min_len = min(lengths)
        max_len = max(lengths)
        if min_len != max_len:
            logging.warning("Not all the trajectories have the same length.")
        for i in range(min_len):
            traj_points = [t[i] for t in trajs]
            traj_points.append(traj_points[0])
            for tp1, tp2 in zip(traj_points[:-1], traj_points[1:]):
                seg_x = [tp1.r[0], tp2.r[0]]
                seg_y = [tp1.r[1], tp2.r[1]]
                seg_z = [tp1.r[2], tp2.r[2]]
                ax.plot(seg_x, seg_y, seg_z, color=(0.2, 0.2, 0.2), linewidth=0.5)

    for i, traj in enumerate(trajs):

        if traj.dim != 3:
            logging.warning(
                "Using plot_3d with a trajectory of %i dimensions"
                " Trajectory No. %i with id %s",
                traj.dim,
                i,
                traj.traj_id,
            )

        # Plotting
        x_data, y_data, z_data = traj.r.x, traj.r.y, traj.r.z

        kwargs["color"] = next(colors)
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

        plt.title(title)
        plt.tick_params(direction="in")
        plt.grid(True)
        ax.set_xlabel(f"x{units}")
        ax.set_ylabel(f"y{units}")
        ax.set_zlabel(f"z{units}")

    if show:
        plt.show()

    return ax


def plot_3D(  # pylint: disable=invalid-name
    trajs: Union[List[Trajectory], Trajectory],
    line_style: str = LINE,
    title: Optional[str] = None,
    legend: bool = True,
    show: bool = True,
    connected: bool = False,
    units: str = "m",
    color=None,
    **kwargs,
):
    """
    .. deprecated:: 0.10.0
        :func:`plot_3D` will be removed in a future version, use
        :func:`plot_3d` instead.

    Plot all the points of trajectories from ``trajs`` in a 3D space.

    Parameters
    ----------
    trajs : Union[List[Trajectory], Trajectory]
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
    warnings.warn(
        "plot_3D is deprecated and will be removed in a future version, "
        "use plot_3d instead",
        DeprecationWarning,
    )
    plot_3d(
        trajs,
        line_style,
        title,
        legend,
        show,
        connected,
        units,
        color,
        **kwargs,
    )


def plot_vs_time(
    trajs: Union[List[Trajectory], Trajectory],
    key: Callable[[Trajectory], Collection[float]],
    line_style: str = LINE,
    x_units: str = "s",
    y_label: Union[str, None] = None,
    title: Optional[str] = None,
    legend: bool = True,
    color=None,
    show: bool = True,
    **kwargs,
):
    if isinstance(trajs, Trajectory):
        trajs = [trajs]

    x_units = "time" + ("" if x_units is None else f" [{x_units}]")

    cycle = itertools.cycle(YUPI_COLORS)
    colors = [next(cycle) for _ in trajs]

    if color is not None:
        if isinstance(color, (str, tuple)):
            kwargs["color"] = color
        elif isinstance(color, list):
            colors = color

    for i, traj in enumerate(trajs):
        if colors is not None:
            if i < len(colors):
                kwargs["color"] = colors[i]
            else:
                kwargs.pop("color")
        y_data = key(traj)
        x_data = traj.t
        traj_id = traj.traj_id if traj.traj_id else f"traj {i}"
        plt.plot(x_data, y_data, line_style, **kwargs, label=traj_id)
        plt.xlabel(x_units)
        if y_label is not None:
            plt.ylabel(y_label)
        plt.grid()
        plt.title(title)

    if legend:
        plt.legend()

    if show:
        plt.show()

    return plt.gca()
