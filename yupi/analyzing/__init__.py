"""
Analyzing package dosctring
"""

from yupi.analyzing.transformations import (
    add_dynamic_reference,
    subsample_trajectory,
    wrap_theta,
    estimate_turning_angles
)
from yupi.analyzing.visualization import plot_trajectories

__all__ = [
    'add_dynamic_reference',
    'plot_trajectories',
    'subsample_trajectory',
    'wrap_theta',
    'estimate_turning_angles'
]