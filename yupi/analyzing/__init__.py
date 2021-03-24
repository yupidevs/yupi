"""
This submodule contains different tools to compute statistical 
funtions for trajectory analysis and visualize the results.

All the resources of this module should be imported directly
from ``yupi.analyzing``. However, we ordered the resources
according the functionality into the following categories.
"""
from yupi.analyzing.processing import (
    turning_angles
)

from yupi.analyzing.transformations import (
    add_dynamic_reference,
    subsample_trajectory,
    wrap_theta
)

from yupi.analyzing.statistics import (
    estimate_turning_angles,
    estimate_msd,
    estimate_kurtosis,
    estimate_vacf,
    estimate_velocity_samples
)
from yupi.analyzing.visualization import (
    plot_trajectories, 
    plot_velocity_hist,
    plot_angle_distribution,
    plot_kurtosis,
    plot_msd,
    plot_vacf
)

__all__ = [
    'add_dynamic_reference',
    'plot_trajectories',
    'plot_velocity_hist',
    'plot_kurtosis',
    'plot_msd',
    'plot_vacf',
    'subsample_trajectory',
    'wrap_theta',
    'estimate_turning_angles',
    'estimate_msd',
    'estimate_kurtosis',
    'estimate_vacf'
]