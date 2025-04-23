"""
This contains specific exceptions related to the library.
"""


class YupiExceptionError(Exception):
    """Generic exception for yupi package"""


class TrajectoryError(YupiExceptionError):
    """Generic exception for handling trajectory errors"""


class LoadTrajectoryError(TrajectoryError):
    """Error while loading a trajectory"""

    def __init__(self, path: str, reason: str = ""):
        self.path = path
        self.message = f"File '{self.path}' is not a valid trajectory"
        if reason:
            self.message += f": {reason}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
