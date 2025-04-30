import pytest

import yupi.tracking as ypt


def test_normal_creation() -> None:
    try:
        ypt.ROI((10, 10), init_mode=ypt.ROI.CENTER_INIT_MODE, scale=0.3)
    except Exception as e:
        pytest.fail(f"ROI creation fails. Exeption: {e}")


def test_invalid_sizes() -> None:
    with pytest.raises(ValueError):
        ypt.ROI((-1, 2))

    with pytest.raises(ValueError):
        ypt.ROI((0.3, 2))


def test_invalid_mode() -> None:
    with pytest.raises(ValueError):
        ypt.ROI((2, 2), "abc")


def test_invalid_scale() -> None:
    with pytest.raises(ValueError):
        ypt.ROI((2, 2), scale=-1)
