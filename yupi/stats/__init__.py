"""
This module contains a set of functions for statistical information
extraction from a trajectory collection.

All the resources of this module should be imported directly
from ``yupi.stats``.
"""

from yupi.stats._stats import (
    collect,
    collect_at_step,
    collect_at_time,
    collect_step_lagged,
    collect_time_lagged,
    kurtosis,
    kurtosis_reference,
    msd,
    msd_ensemble,
    msd_time,
    psd,
    speed_ensemble,
    turning_angles_ensemble,
    vacf,
    vacf_ensemble,
    vacf_time,
)

__all__ = [
    "turning_angles_ensemble",
    "collect_at_step",
    "collect_at_time",
    "collect_step_lagged",
    "collect_time_lagged",
    "collect",
    "speed_ensemble",
    "msd_ensemble",
    "msd_time",
    "vacf_ensemble",
    "msd",
    "vacf_time",
    "vacf",
    "kurtosis",
    "kurtosis_reference",
    "psd",
]
