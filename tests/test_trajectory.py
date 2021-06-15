import os
import numpy as np
import pytest
from yupi import Trajectory


def test_creation_by_xyz():
    Trajectory(x=[1, 2])
    Trajectory(x=[1, 2], y=[2, 3])
    Trajectory(x=[1, 2], y=[2, 3], z=[1, 4])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2], z=[2, 5])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2, 6], z=[5])

    with pytest.raises(ValueError):
        Trajectory(x=[2], y=[2, 6], z=[5, 8])


def test_creation_by_dimensions():
    Trajectory(dimensions=[[1, 2]])
    Trajectory(dimensions=[[1, 2], [2, 3]])
    Trajectory(dimensions=[[1, 2], [2, 3], [1, 4]])
    Trajectory(dimensions=[[1, 2], [2, 3], [1, 4], [7, 8]])

    with pytest.raises(ValueError):
        Trajectory(dimensions=[[1, 2], [2]])

    with pytest.raises(ValueError):
        Trajectory(dimensions=[[1, 2], [2], [2, 5]])

    with pytest.raises(ValueError):
        Trajectory(dimensions=[[1, 2], [2, 6], [5]])

    with pytest.raises(ValueError):
        Trajectory(dimensions=[[2], [2, 6], [5, 8]])


def test_creation_by_points():
    Trajectory(points=[[1, 2]])
    Trajectory(points=[[1, 2], [2, 3]])
    Trajectory(points=[[1, 2, 4], [2, 3, 2], [1, 4, 8]])
    Trajectory(points=[[1, 2, 7, 3], [2, 3, 5, 3]])

    with pytest.raises(ValueError):
        Trajectory(points=[[1, 2], [2]])

    with pytest.raises(ValueError):
        Trajectory(points=[[1, 2], [2], [2, 5]])


def test_creation_with_time():
    Trajectory(x=[1, 2], y=[2, 3], t=[0, 0.1])
    Trajectory(x=[1, 2], y=[2, 3], dt=0.1)
    Trajectory(x=[1, 2], y=[2, 3], t=[0, 0.1], dt=0.1)
    Trajectory(x=[1, 2], y=[2, 3], t=[0.4, 0.5], dt=0.1, t0=0.4)

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2, 3], t=[0])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2, 3], t=[0, 0.1], dt=0.2)

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2, 3], t=[0.4, 0.5], dt=0.1)

def test_creation_with_ang():
    Trajectory(x=[1, 2], y=[2, 3], ang=[0, 0.1])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2, 3], ang=[0.1])


def test_creation_general():
    Trajectory(x=[1, 2], y=[2, 3], t=[0, 1], ang=[0, 0], traj_id='test')
    Trajectory(x=[1, 2], y=[2, 3], dt=0.5, ang=[0, 1.2], traj_id='test')
    Trajectory(points=[[1, 2], [2, 3]], dt=0.5, ang=[0, 1.2], traj_id='test')
    Trajectory(dimensions=[[1, 2], [2, 3]], dt=0.5, t=[1, 1.5], t0=1,
               traj_id='test')


def test_iteration():
    points = np.array([[1.0, 2.0], [4.0, 3.0]])
    angs = np.array([0.0, 2.0])
    time = np.array([0.0, 1.0])
    t1 = Trajectory(points=points, ang=angs, t=time)

    for i, tp in enumerate(t1):
        point = points[i]
        ang = angs[i]
        t = time[i]

        # Position
        assert point[0] == tp.r[0]
        assert point[1] == tp.r[1]

        # Time
        assert t == tp.t

        # Ang
        assert ang == tp.ang


def test_save():
    t1 = Trajectory(x=[1,2,3], y=[4,5,6])

    # Wrong trajectory file extension at saving
    with pytest.raises(ValueError):
        t1.save('t1', file_type='abc')

    # Saving json
    t1.save('t1', file_type='json')

    # Saving csv
    t1.save('t1', file_type='csv')
    

def test_load():
    # Wrong trajectory file extension at loading
    with pytest.raises(ValueError):
        t1 = Trajectory.load('t1.abc')

    # Loading json
    t1 = Trajectory.load('t1.json')
    for tp, point in zip(t1, [[1,4], [2,5], [3,6]]):
        assert (np.array(point) == tp.r).all()

    # Loading csv
    t1 = Trajectory.load('t1.csv')
    for tp, point in zip(t1, [[1,4], [2,5], [3,6]]):
        assert (np.array(point) == tp.r).all()

    os.remove('t1.json')
    os.remove('t1.csv')
