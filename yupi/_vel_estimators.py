import enum

from yupi.vector import Vector


class VelMethod(enum.Enum):
    """
    Enum to define the method to calculate the velocity.
    """

    EULER = enum.auto()
    CENTERED = enum.auto()
    WIN_CENTERED = enum.auto()


def euler(traj, **kwargs) -> Vector:
    return (traj.r[1:] - traj.r[:-1]) / (traj.t[1:] - traj.t[:-1])


def centered(traj, **kwargs) -> Vector:
    return (traj.r[2:] - traj.r[:-2]) / (traj.t[2:] - traj.t[:-2])


FUNCTIONS = {
    VelMethod.EULER: euler,
    VelMethod.CENTERED: centered,
}
