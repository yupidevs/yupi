"""
This submodule contains different tools to compute statistical
funtions for trajectory analysis and visualize the results.

All the resources of this module should be imported directly
from ``yupi.analyzing``. However, we ordered the resources
according the functionality into the following categories.
"""

from yupi.stats._stats import (
    turning_angles_ensemble,
    speed_ensemble,
    msd_ensemble,
    msd_time,
    vacf_ensemble,
    msd,
    vacf_time,
    vacf,
    kurtosis,
    kurtosis_reference,
    psd
)


__all__ = [
    'turning_angles_ensemble',
    'speed_ensemble',
    'msd_ensemble',
    'msd_time',
    'vacf_ensemble',
    'msd',
    'vacf_time',
    'vacf',
    'kurtosis',
    'kurtosis_reference',
    'psd'
]
