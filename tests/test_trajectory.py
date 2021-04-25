import os
import pytest
import yupi

def test_creation():
    try:
        yupi.Trajectory(x=[1.0, 2.0], y=[2.0, 3.0])
    except Exception as e:
        pytest.fail(f'Trajectory creation fails. Exeption: {e}')

    try:
        yupi.Trajectory(
            x=[1.0, 2.0],
            y=[2.0, 3.0],
            z=[1.0, 3.0],
            t=[0.0, 1.0],
            ang=[0.0,0.0],
            id='test'
        )
    except Exception as e:
        pytest.fail(f'Trajectory creation fails. Exeption: {e}')
    

def test_iteration():
    t1 = yupi.Trajectory(x=[1.0, 2.0], y=[2.0, 3.0])
    tps = [tp[:2] for tp in t1]

    assert len(tps) > 0
    assert tps == [[1,2],[2,3]]

def test_save():
    t1 = yupi.Trajectory(x=[1.0,2.0,3.0])

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
    except Exception as e:
        pytest.fail(f'Trajectory json load fails. Exeption: {e}')

    # Loading csv
    try:
        t1 = yupi.Trajectory.load('t1.csv')
    except Exception as e:
        pytest.fail(f'Trajectory csv load fails. Exeption: {e}')

    os.remove('t1.json')
    os.remove('t1.csv')