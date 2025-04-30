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
)
from yupi.stats.kurtosis import KurtosisStat, KurtosisTimeAvgStat, kurtosis_reference
from yupi.stats.msd import MsdStat, MsdTimeAvgStat
from yupi.stats.psd import PsdStat
from yupi.stats.speed import SpeedStat
from yupi.stats.turning_angles import TurningAngleStat, turning_angles
from yupi.stats.vacf import VacfStat, VacfTimeAvgStat

__all__ = [
    "KurtosisStat",
    "KurtosisTimeAvgStat",
    "MsdStat",
    "MsdTimeAvgStat",
    "PsdStat",
    "PsdStatKurtosisStat",
    "SpeedStat",
    "TurningAngleStat",
    "TurningAngleStat",
    "VacfStat",
    "VacfStat",
    "VacfTimeAvgStat",
    "VacfTimeAvgStat",
    "collect",
    "collect_at_step",
    "collect_at_time",
    "collect_step_lagged",
    "collect_time_lagged",
    "kurtosis_reference",
    "turning_angles",
]
