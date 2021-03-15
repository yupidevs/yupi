"""
Analyzing package dosctring
"""

from yupi.analyzing.transformations import (
    add_dynamic_reference,
    subsample_trajectory,
    wrap_theta
)

from yupi.analyzing.statistics import (
    estimate_turning_angles,
    estimate_msd,
    estimate_kurtosis,
    estimate_vacf
)
from yupi.analyzing.visualization import plot_trajectories

__all__ = [
    'add_dynamic_reference',
    'plot_trajectories',
    'subsample_trajectory',
    'wrap_theta',
    'estimate_turning_angles',
    'estimate_msd',
    'estimate_kurtosis',
    'estimate_vacf'
]