"""
This contains styling utilities for the library plots.
"""

from functools import wraps
from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.projections import PolarAxes

LINE = "-"
DOTTED = "o"
LINE_DOTTED = "-o"

# Main colors
BLUE = "#3489b1"
MAGENTA = "#a14383"
YELLOW = "#e5d935"
RED = "#af3c3c"
GREEN = "#54ac43"
ORANGE = "#e88d26"
MID_BLUE = "#3c4baf"

#  Light colors
LIGHT_BLUE = "#99d2ec"
LIGHT_MAGENTA = "#eaa0d2"
LIGHT_YELLOW = "#fdf584"
LIGHT_RED = "#ea8080"
LIGHT_GREEN = "#a6ec98"
LIGHT_ORANGE = "#f7c790"

YUPI_COLORS: list[Any] = [BLUE, MAGENTA, YELLOW, RED, GREEN, ORANGE, MID_BLUE]

YUPI_LIGHT_COLORS = [
    LIGHT_BLUE,
    LIGHT_MAGENTA,
    LIGHT_YELLOW,
    LIGHT_RED,
    LIGHT_GREEN,
    LIGHT_ORANGE,
]


def _plot_basic_properties(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(
        *args: Any,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        grid: bool = False,
        show: bool = True,
        legend: bool = False,
        xscale: str | None = None,
        yscale: str | None = None,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        **kwargs: Any,
    ) -> Axes | PolarAxes:
        ax = func(*args, **kwargs)
        plt.grid(grid)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if legend:
            plt.legend()
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if xscale is not None:
            plt.xscale(xscale)
        if yscale is not None:
            plt.yscale(yscale)
        if show:
            plt.show()
        return ax

    return wrapper
