"""
This module contains a set of functions for statistical information
extraction from a trajectory collection.

All the resources of this module should be imported directly
from ``yupi.stats``.
"""

from yupi.stats._stats import (
    collect_at_step,
    collect_at_time,
    collect_step_lagged,
    collect_time_lagged,
    collect,
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
    'collect_at_step',
    'collect_at_time',
    'collect_step_lagged',
    'collect_time_lagged',
    'collect',
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
