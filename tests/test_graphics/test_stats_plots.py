import numpy as np
import pytest
from matplotlib import pyplot as plt
from pytest import MonkeyPatch

from yupi.graphics import (
    plot_angles_hist,
    plot_hist,
    plot_hists,
    plot_kurtosis,
    plot_msd,
    plot_psd,
    plot_speed_hist,
    plot_vacf,
)
from yupi.stats import kurtosis, kurtosis_reference, msd, psd, vacf
from yupi.trajectory import Trajectory


@pytest.fixture
def trajs() -> list[Trajectory]:
    t1 = Trajectory(x=[0, 0, 4], y=[0, 3, 3])
    t2 = Trajectory(x=[4, 7, 7], y=[4, 4, 8])
    t3 = Trajectory(x=[1, 2, 3], y=[2, 3, 6])
    return [t1, t2, t3]


def test_plot_hist(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    ax = plot_hist([1, 2, 3])
    assert isinstance(ax, plt.Axes)


def test_plot_hists(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_hists([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Test with multiple histograms
    plot_hists([[1, 2, 3], [3, 4, 5]])

    with pytest.raises(ValueError):
        # Test with empty histograms
        plot_hists([[1, 2, 3], [3, 4, 5]], [{}])

    plot_hists([[1, 2, 3], [3, 4, 5]], labels=["A", "B"], histtype="step", filled=True)

    plot_hists(
        [[1, 2, 3], [3, 4, 5]],
        labels=["A", "B"],
        histtype="step",
        filled=True,
        legend=True,
        xlim=(0, 10),
        ylim=(0, 10),
        xscale="log",
        yscale="log",
        title="Title",
    )


def test_plot_speed_hist(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_speed_hist(np.array([1, 2, 3]), units="ns/s", density=True)


def test_plot_angles_hist(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    ax = plt.gca()
    with pytest.raises(ValueError):
        plot_angles_hist(np.array([1, 2, 3]), 36, density=True, ax=ax)

    plot_angles_hist(np.array([1, 2, 3]), 36, density=True)

    ax = plt.axes(projection="polar")
    plot_angles_hist(np.array([1, 2, 3]), 36, density=True, ax=ax)
    plt.axes()  # Reset to default axes


def test_plot_msd(monkeypatch: MonkeyPatch, trajs: list[Trajectory]) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    msd_mean, msd_std = msd(trajs, lag=1)
    plot_msd(msd_mean, msd_std, dt=1, lag=1, x_units="some_unit", y_units="some_unit")


def test_plot_psd(monkeypatch: MonkeyPatch, trajs: list[Trajectory]) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    psd_mean, psd_std, frec = psd(trajs, lag=1)
    plot_psd(psd_mean, frec, psd_std)


def test_plot_vacf(monkeypatch: MonkeyPatch, trajs: list[Trajectory]) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    vacf_mean, _ = vacf(trajs, lag=1)
    plot_vacf(vacf_mean, dt=1, lag=1, x_units="some_unit", y_units="some_unit")


def test_plot_kurtosis(monkeypatch: MonkeyPatch, trajs: list[Trajectory]) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    kurt_ref = kurtosis_reference(trajs)
    kurt_mean, _ = kurtosis(trajs, lag=1)
    plot_kurtosis(
        kurt_mean,
        kurtosis_ref=kurt_ref,
    )
    plot_kurtosis(
        kurt_mean,
        kurtosis_ref=kurt_ref,
        dt=1,
        units="s",
    )
