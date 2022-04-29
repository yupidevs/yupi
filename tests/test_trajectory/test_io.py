import os

import numpy as np
import pytest

from yupi import Trajectory


def test_save():
    x = [1, 2, 3]
    y = [4, 5, 6]
    t1 = Trajectory(x=x, y=y)

    # Wrong trajectory file extension at saving
    with pytest.raises(ValueError):
        t1.save("t1", file_type="abc")

    t1.save("t1", file_type="json")
    t1.save("t1", file_type="csv")

    t2 = Trajectory(x=x, y=y, t=[0.0, 0.5, 2.0])
    t2.save("t2", file_type="json")
    t2.save("t2", file_type="csv")


def test_load():
    # Wrong trajectory file extension at loading
    with pytest.raises(ValueError):
        t1 = Trajectory.load("t1.abc")

    # Loading json
    t1 = Trajectory.load("t1.json")
    for tp, point in zip(t1, [[1, 4], [2, 5], [3, 6]]):
        assert (np.array(point) == tp.r).all()
    t2 = Trajectory.load("t2.json")

    # Loading csv
    t1 = Trajectory.load("t1.csv")
    for tp, point in zip(t1, [[1, 4], [2, 5], [3, 6]]):
        assert (np.array(point) == tp.r).all()
    t2 = Trajectory.load("t2.csv")

    os.remove("t1.json")
    os.remove("t1.csv")
    os.remove("t2.json")
    os.remove("t2.csv")
