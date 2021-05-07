import os
import numpy as np
import pytest
import yupi

def test_creation():
    try:
        yupi.Trajectory(x=[1, 2], y=[2, 3])
    except Exception as e:
        pytest.fail(f'Trajectory creation fails. Exeption: {e}')

    try:
        yupi.Trajectory(dimensions=[[1, 2],[2, 3]])
    except Exception as e:
        pytest.fail(f'Trajectory creation fails. Exeption: {e}')

    try:
        yupi.Trajectory(points=[[1, 2],[2, 3]])
    except Exception as e:
        pytest.fail(f'Trajectory creation fails. Exeption: {e}')

    try:
        yupi.Trajectory(
            x=[1, 2],
            y=[2, 3],
            z=[1, 3],
            t=[0, 1],
            ang=[0,0],
            traj_id='test'
        )
    except Exception as e:
        pytest.fail(f'Trajectory creation fails. Exeption: {e}')
    

def test_iteration():
    t1 = yupi.Trajectory(x=[1, 4], y=[2, 3])
    for tp, point in zip(t1, [[1,2],[4,3]]):
        assert (np.array(point) == tp.r).all()

def test_save():
    t1 = yupi.Trajectory(x=[1,2,3], y=[4,5,6])

    # Wrong trajectory file extension at saving
    with pytest.raises(ValueError):
        t1.save('t1', file_type='abc')

    # Saving json
    try:
        t1.save('t1', file_type='json')
    except Exception as e:
        pytest.fail(f'Trajectory json save fails. Exeption: {e}')

    # Saving csv
    try:
        t1.save('t1', file_type='csv')
    except Exception as e:
        pytest.fail(f'Trajectory csv save fails. Exeption: {e}')
    

def test_load():
    # Wrong trajectory file extension at loading
    with pytest.raises(ValueError):
        t1 = yupi.Trajectory.load('t1.abc')

    # Loading json
    try:
        t1 = yupi.Trajectory.load('t1.json')
        for tp, point in zip(t1, [[1,4], [2,5], [3,6]]):
            assert (np.array(point) == tp.r).all()
    except Exception as e:
        pytest.fail(f'Trajectory json load fails. Exeption: {e}')

    # Loading csv
    try:
        t1 = yupi.Trajectory.load('t1.csv')
        for tp, point in zip(t1, [[1,4], [2,5], [3,6]]):
            assert (np.array(point) == tp.r).all()
    except Exception as e:
        pytest.fail(f'Trajectory csv load fails. Exeption: {e}')

    os.remove('t1.json')
    os.remove('t1.csv')