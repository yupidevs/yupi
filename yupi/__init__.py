"""
All the resources of the library should be imported directly
from one of the aforementioned modules.
"""

import logging
import warnings

import yupi.core
import yupi.generators
import yupi.graphics
import yupi.stats
import yupi.tracking
import yupi.transformations
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
    "Trajectory",
    "TrajectoryPoint",
    "Features",
    "Vector",
    "DiffMethod",
    "WindowType",
]

__version__ = "0.12.2"
