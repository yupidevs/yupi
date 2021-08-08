"""
This module contains a set of functions for statistical information
extraction from a trajectory collection.

All the resources of this module should be imported directly
from ``yupi.stats``.
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
