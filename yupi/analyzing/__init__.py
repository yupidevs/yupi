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
    estimate_velocity_samples,
    estimate_msd_ensemble,
    estimate_msd_time,
    estimate_msd,
    estimate_vacf_ensemble,
    estimate_vacf_time,
    estimate_vacf,
    estimate_kurtosis_ensemble,
    estimate_kurtosis_time,
    estimate_kurtosis
)

from yupi.analyzing.visualization import (
    plot_trajectories,
    plot_trajectory,
    plot_velocity_hist,
    plot_angle_distribution,
    plot_kurtosis,
    plot_msd,
    plot_vacf
)

__all__ = [
    'turning_angles',
    'add_dynamic_reference',
    'plot_trajectories',
    'plot_trajectory',
    'plot_velocity_hist',
    'plot_angle_distribution',
    'plot_kurtosis',
    'plot_msd',
    'plot_vacf',
    'subsample_trajectory',
    'wrap_theta',
    'estimate_turning_angles',
    'estimate_velocity_samples',
    'estimate_msd_ensemble',
    'estimate_msd_time',
    'estimate_msd',
    'estimate_vacf_ensemble',
    'estimate_vacf_time',
    'estimate_vacf',
    'estimate_kurtosis_ensemble',
    'estimate_kurtosis_time',
    'estimate_kurtosis'
]
