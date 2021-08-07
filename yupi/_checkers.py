from typing import List
import numpy as np
from yupi.trajectory import Trajectory, _threshold

def _check_uniform_time_spaced(func):
    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if any((abs(t.dt_std - 0) > _threshold for t in trajs)):
            raise ValueError('All trajectories must be uniformly time spaced')
        return func(trajs, *args, **kwargs)
    wrapper.__doc__ = func.__doc__
    return wrapper

def _check_same_dt(func):
    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if trajs:
            dt = trajs[0].dt
            if any((abs(t.dt - dt) > _threshold for t in trajs)):
                raise ValueError("All trajectories must have the same 'dt'")
        return func(trajs, *args, **kwargs)
    wrapper.__doc__ = func.__doc__
    return wrapper

def _check_same_dim(func):
    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if trajs:
            dim = trajs[0].dim
            if any((t.dim != dim for t in trajs)):
                raise ValueError("All trajectories must have the same dimensions")
        return func(trajs, *args, **kwargs)
    wrapper.__doc__ = func.__doc__
    return wrapper

def _check_exact_dim(dim):
    def _check_exact_dim_decorator(func):
        def wrapper(trajs: List[Trajectory], *args, dim=dim, **kwargs):
            if any((t.dim != dim for t in trajs)):
                raise ValueError(f"All trajectories must be {dim}-dimensional")
            return func(trajs, *args, **kwargs)
        wrapper.__doc__ = func.__doc__
        return wrapper
    return _check_exact_dim_decorator

def _check_same_r0(func):
    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if trajs:
            r0 = trajs[0].r[0]
            if any((abs(t.r[0] - r0) > _threshold for t in trajs)):
                raise ValueError("All trajectories must have the same initial position")
        return func(trajs, *args, **kwargs)
    wrapper.__doc__ = func.__doc__
    return wrapper

def _check_same_lenght(func):
    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if trajs:
            length = len(trajs[0])
            if any((abs(len(t) - length) > _threshold for t in trajs)):
                raise ValueError("All trajectories must have the same length")
        return func(trajs, *args, **kwargs)
    wrapper.__doc__ = func.__doc__
    return wrapper

def _check_same_t(func):
    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if trajs:
            zipped_trajs = zip(trajs[:-1], trajs[1:])
            if not all((np.allclose(t0.t, t1.t, atol=_threshold) for t0, t1 in zipped_trajs)):
                raise ValueError("All trajectories must have the same 't' values")
        return func(trajs, *args, **kwargs)
    wrapper.__doc__ = func.__doc__
    return wrapper
