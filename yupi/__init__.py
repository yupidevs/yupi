"""
All the resources of the library should be imported directly
from one of the aforementioned modules.
"""

import logging
import warnings

from yupi._differentiation import DiffMethod, WindowType
from yupi.features import Features
from yupi.trajectory import Trajectory, TrajectoryPoint
from yupi.vector import Vector

warnings.filterwarnings("default", category=DeprecationWarning, module="yupi")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

__all__ = [
    "Trajectory",
    "TrajectoryPoint",
    "Features",
    "Vector",
    "DiffMethod",
    "WindowType",
]

__version__ = "0.11.1"
