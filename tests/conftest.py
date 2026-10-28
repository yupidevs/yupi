from collections.abc import Iterator

import pytest
from matplotlib import pyplot as plt


@pytest.fixture
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")
