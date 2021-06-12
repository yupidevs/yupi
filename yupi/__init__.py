"""
The API of yupi is divided into four modules:
 *  yupi: General classes defining concepts used by all the other modules
 *  yupi.analyzing: Statistical methods and visualization tools to analize trajectories
 *  yupi.generating: Models to generate trajectories with given statistical constrains
 *  yupi.tracking: Tools to extract trajectories from image sequences based on the color or the motion of the object
    being tracked

All the resources of the library should be imported directly
from one of the aforementioned modules. However, we ordered the resources
according the functionality into subcategories to simplify searching.

"""

import logging
from yupi.trajectory import Trajectory
from yupi.vector import Vector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

__all__ = [
    'Trajectory',
    'Vector'
]

__version__ = '0.4.2'