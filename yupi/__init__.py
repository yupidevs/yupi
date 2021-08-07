"""
The API of yupi is divided into seven modules:
  * yupi: General classes defining concepts used by all the other modules
  * yupi.tracking: Tools to extract trajectories from image sequences
  * yupi.generators: Models to generate trajectories with given statistical constrains.
  * yupi.transformations: Tools to transform trajectories (resamplers, filters, etc.).
  * yupi.estimators: Tools to exctract information from trajectories
  * yupi.stats: Tools to extract statistical data from a set of trajectories.
  * yupi.graphics: Tools for visualizing trajectories and statistical data.
    based on the color or the motion of the object being tracked.

All the resources of the library should be imported directly
from one of the aforementioned modules.
"""

import logging
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
    'Vector'
]

__version__ = '0.6.1'
