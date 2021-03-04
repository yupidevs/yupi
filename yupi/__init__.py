"""
All package docstring
"""

from yupi.trajectory import Trajectory
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

__all__ = [
    'Trajectory'
]

__version__ = '0.2.0'