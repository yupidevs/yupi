import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex, to_rgba
from matplotlib.patches import Rectangle
from matplotlib.projections.polar import PolarAxes

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
from yupi.graphics._style import BLUE, GREEN, LIGHT_GREEN, MAGENTA, RED, YUPI_LIGHT_COLORS

pytestmark = pytest.mark.usefixtures("close_figures")


@pytest.fixture
def values() -> np.ndarray:
    return np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0])


def test_plot_hist_uses_the_first_light_color(values: np.ndarray) -> None:
    ax = plot_hist(values, show=False)

    assert ax.patches
    assert to_hex(ax.patches[0].get_facecolor()) == YUPI_LIGHT_COLORS[0]


def test_plot_hist_keeps_a_given_color(values: np.ndarray) -> None:
    ax = plot_hist(values, color=RED, show=False)

    assert to_hex(ax.patches[0].get_facecolor()) == RED


def test_plot_hists_draws_one_outline_per_value_set(values: np.ndarray) -> None:
    ax = plot_hists([values, values + 1], labels=["a", "b"], show=False)

    assert [patch.get_label() for patch in ax.patches] == ["a", "b"]
    assert [to_hex(patch.get_edgecolor()) for patch in ax.patches] == [BLUE, MAGENTA]


def test_plot_hists_filled_adds_a_filled_copy(values: np.ndarray) -> None:
    ax = plot_hists([values, values + 1], filled=True, show=False)

    assert len(ax.patches) == 4


def test_plot_hists_rejects_mismatched_kwargs(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="length of 'kwargs_list'"):
        plot_hists([values, values], kwargs_list=[{}], show=False)


def test_plot_speed_hist_is_a_density_with_units(values: np.ndarray) -> None:
    ax = plot_speed_hist(values, show=False, density=False)

    bars = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    area = sum(bar.get_height() * bar.get_width() for bar in bars)
    assert area == pytest.approx(1)
    assert ax.get_xlabel() == "speed [m/s]"
    assert ax.get_ylabel() == "pdf"


@pytest.mark.parametrize(("units", "label"), [("km/h", "speed [km/h]"), (None, "speed")])
def test_plot_speed_hist_labels_the_units(values: np.ndarray, units: str, label: str) -> None:
    ax = plot_speed_hist(values, show=False, units=units)

    assert ax.get_xlabel() == label


def test_plot_angles_hist_draws_on_polar_axes() -> None:
    angles = np.linspace(-np.pi, np.pi, 50)

    ax = plot_angles_hist(angles, bins=10, show=False)

    assert isinstance(ax, PolarAxes)
    assert len(ax.patches) == 10
    assert ax.get_theta_offset() == pytest.approx(np.pi / 2)


def test_plot_angles_hist_rejects_cartesian_axes() -> None:
    with pytest.raises(ValueError, match="The axes must be polar"):
        plot_angles_hist(np.zeros(3), bins=3, show=False, ax=plt.axes())


def test_plot_msd_plots_lag_times_on_log_axes() -> None:
    msd = np.array([1.0, 2.0, 4.0, 8.0])

    ax = plot_msd(msd, msd / 10, dt=0.5, lag=4, show=False)

    x, y = ax.lines[0].get_data()
    assert x == pytest.approx([0.0, 0.5, 1.0, 1.5])
    assert y == pytest.approx(msd)
    assert len(ax.collections) == 1
    assert (ax.get_xscale(), ax.get_yscale()) == ("log", "log")
    assert ax.get_xlabel() == "lag time [s]"
    assert ax.get_ylabel() == r"$\mathrm{msd \; [m^2]}$"


def test_plot_kurtosis_with_dt_spreads_time_and_shades_the_reference() -> None:
    kurtosis = np.array([3.0, 2.5, 2.0])

    ax = plot_kurtosis(kurtosis, dt=1.0, kurtosis_ref=3.0, show=False)

    x, _ = ax.lines[0].get_data()
    assert x == pytest.approx([0.0, 1.5, 3.0])
    assert ax.lines[0].get_color() == GREEN
    assert ax.collections[0].get_facecolor()[0] == pytest.approx(to_rgba(LIGHT_GREEN))
    assert ax.get_xlabel() == "time [s]"
    assert ax.get_ylabel() == "kurtosis"


def test_plot_kurtosis_without_time_plots_against_the_index() -> None:
    ax = plot_kurtosis(np.array([3.0, 2.5, 2.0]), show=False)

    x, _ = ax.lines[0].get_data()
    assert x == pytest.approx([0, 1, 2])
    assert ax.get_xlabel() == ""
    assert not ax.collections


def test_plot_kurtosis_uses_a_given_time_array() -> None:
    t_array = np.array([0.0, 2.0, 5.0])

    ax = plot_kurtosis(np.array([3.0, 2.5, 2.0]), t_array=t_array, show=False)

    x, _ = ax.lines[0].get_data()
    assert x == pytest.approx(t_array)


def test_plot_vacf_adds_a_log_inset() -> None:
    vacf = np.array([1.0, 0.5, 0.25])

    ax = plot_vacf(vacf, dt=0.1, lag=3, show=False)

    x, _ = ax.lines[0].get_data()
    assert x == pytest.approx([0.0, 0.1, 0.2])
    assert ax.lines[0].get_color() == RED
    inset = next(other for other in ax.figure.axes if other is not ax)
    assert inset.get_yscale() == "log"


def test_plot_vacf_without_inset_draws_one_axes() -> None:
    ax = plot_vacf(np.array([1.0, 0.5]), dt=1, lag=2, log_inset=False, show=False)

    assert ax.figure.axes == [ax]


def test_plot_psd_shades_the_deviation() -> None:
    frec = np.array([1.0, 10.0, 100.0])
    psd = np.array([5.0, 3.0, 1.0])

    ax = plot_psd(psd, frec, psd_std=psd / 10, show=False)

    assert len(ax.collections) == 1
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["psd", "psd_std"]
    assert ax.get_xlabel() == "frequency [rad/s]"
    assert (ax.get_xscale(), ax.get_yscale()) == ("log", "log")


def test_plot_psd_in_hertz() -> None:
    ax = plot_psd(np.array([1.0, 2.0]), np.array([1.0, 2.0]), omega=False, show=False)

    assert ax.get_xlabel() == "frequency [Hz]"
    assert not ax.collections
