from yupi import Trajectory

def subsample(traj: Trajectory, step=1, step_in_seconds=False):
    """
    Sample the trajectory ``traj`` by removing evenly spaced
    points according to ``step``.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory.
    step : int, optional
        Number of sample points or period, depending on the value
        of ``step_in_seconds``. By default 1.
    step_in_seconds : bool, optional
        If True, ``step`` is considered as the number of sample
        points. Otherwise, ``step`` is interpreted as the sample
        period, in seconds. By default False.

    Returns
    -------
    Trajectory
        Output trajectory.
    """

    if step_in_seconds:
        step = int(step / traj.dt)

    points = traj.r[::step]
    ang = traj.ang[::step] if traj.ang is not None else None
    t = traj.t[::step] if traj.t is not None else None
    return Trajectory(points=points, t=t, ang=ang, dt=step*traj.dt)
