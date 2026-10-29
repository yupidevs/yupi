import numpy as np
import pytest
from matplotlib import pyplot as plt

from yupi.graphics import plot_hist

pytestmark = pytest.mark.usefixtures("close_figures")


def test_basic_properties_are_applied_to_the_plot() -> None:
    ax = plot_hist(
        np.array([1.0, 2.0, 3.0]),
        title="Title",
        xlabel="x",
        ylabel="y",
        xlim=(0, 5),
        ylim=(0, 2),
        xscale="symlog",
        yscale="log",
        label="values",
        legend=True,
        show=False,
    )

    assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ("Title", "x", "y")
    assert ax.get_xlim() == pytest.approx((0, 5))
    assert ax.get_ylim() == pytest.approx((0, 2))
    assert (ax.get_xscale(), ax.get_yscale()) == ("symlog", "log")
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["values"]


def test_basic_properties_show_the_plot_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))

    plot_hist(np.array([1.0, 2.0]))
    plot_hist(np.array([1.0, 2.0]), show=False)

    assert shown == [True]
