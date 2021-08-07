from yupi import Trajectory

def subsample(traj: Trajectory, step=1, new_traj_id: str = None):
    """
    Sample the trajectory ``traj`` by removing evenly spaced
    points according to ``step``.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory.
    step : int, optional
        Number of sample points or period. By default 1.
    Returns
    -------
    Trajectory
        Output trajectory.
    """

    points = traj.r[::step]
    ang = traj.ang[::step] if traj.ang is not None else None
    t = traj.t[::step] if traj.t is not None else None

    subsampled_traj = Trajectory(
        points=points,
        t=t,
        ang=ang,
        dt=step*traj.dt,
        traj_id=new_traj_id
    )
    return subsampled_traj
