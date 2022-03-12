"""
All the resources of the library should be imported directly
from one of the aforementioned modules.
"""

import logging
from yupi.features import Features
from yupi.trajectory import Trajectory, TrajectoryPoint
from yupi.vector import Vector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

__all__ = [
    'Trajectory',
    'TrajectoryPoint',
    'Features',
    'Vector'
]

__version__ = '0.8.5'
