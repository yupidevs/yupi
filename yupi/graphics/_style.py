"""
This contains styling utilities for the library plots.
"""

from typing import Optional

import matplotlib.pyplot as plt

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

YUPI_COLORS = [BLUE, MAGENTA, YELLOW, RED, GREEN, ORANGE, MID_BLUE]

YUPI_LIGHT_COLORS = [
    LIGHT_BLUE,
    LIGHT_MAGENTA,
    LIGHT_YELLOW,
    LIGHT_RED,
    LIGHT_GREEN,
    LIGHT_ORANGE,
]


def _plot_basic_properties(func):
    def wrapper(
        *args,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        grid: bool = False,
        show: bool = True,
        legend: bool = False,
        xscale: Optional[str] = None,
        yscale: Optional[str] = None,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        **kwargs,
    ):
        func(*args, **kwargs)
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

    wrapper.__doc__ = func.__doc__
    return wrapper
