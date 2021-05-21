

class YupiException(Exception):
    """Generic exception for yupi package"""

class TrajectoryError(YupiException):
    """Generic exception for handling trajectory errors"""

class LoadTrajectoryError(TrajectoryError):
    """Error while loading a trajectory"""

    def __init__(self, path):
        self.path = path
        self.message = f"File: '{self.path}' is not a valid trajectory"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
