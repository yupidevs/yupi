"""
All the resources of the library should be imported directly
from one of the aforementioned modules.
"""

import logging
import warnings

from yupi._checkers import (
    DifferentDimensionError,
    DifferentDtError,
    DifferentLengthError,
    DifferentTimeVectorError,
    NotUniformTimeSpacedError,
)
from yupi._differentiation import DiffMethod, WindowType
from yupi.trajectory import Trajectory, TrajectoryPoint
from yupi.vector import Vector

warnings.filterwarnings("default", category=DeprecationWarning, module="yupi")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

__all__ = [
    "DiffMethod",
    "DifferentDimensionError",
    "DifferentDtError",
    "DifferentLengthError",
    "DifferentTimeVectorError",
    "NotUniformTimeSpacedError",
    "Trajectory",
    "TrajectoryPoint",
    "Vector",
    "WindowType",
]

__version__ = "0.13.0"
