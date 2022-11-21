"""
This module contains a set of functions capable of applying
transformations to trajectories such as filtering, resampling, etc.

All the resources of this module should be imported directly
from ``yupi.transormations``.
"""

from yupi.transformations._filters import exp_convolutional_filter
from yupi.transformations._resamplers import resample, subsample
from yupi.transformations._transformations import add_moving_FoR

__all__ = [
    "subsample",
    "exp_convolutional_filter",
    "add_moving_FoR",
    "resample",
]
