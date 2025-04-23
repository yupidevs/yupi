"""
This contains specific exceptions related to the library.
"""

from typing import Any

from yupi.trajectory import Trajectory


class YupiExceptionError(Exception):
    """Generic exception for yupi package"""


class TrajectoryError(YupiExceptionError):
    """Generic exception for handling trajectory errors"""

    def __init__(self, traj: Trajectory, *args: object, **kwargs: Any):
        self.traj = traj
        super().__init__(*args, **kwargs)


class TrajectoryGroupError(YupiExceptionError):
    """Generic exception for handling errors in a collection of trajectories"""

    def __init__(self, trajs: list[Trajectory], *args: object, **kwargs: Any):
        self.trajs = trajs
        super().__init__(*args, **kwargs)


class LoadTrajectoryError(YupiExceptionError):
    """Error while loading a trajectory"""

    def __init__(self, path: str, reason: str = ""):
        self.path = path
        self.message = f"File '{self.path}' is not a valid trajectory"
        if reason:
            self.message += f": {reason}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
