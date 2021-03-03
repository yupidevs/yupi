"""
Tracking package docstring
"""

from yupi.tracking.trackers import (
    ROI,
    ObjectTracker,
    CameraTracker,
    TrackingScenario
)
from yupi.tracking.algorithms import (
    Algorithm,
    IntensityMatching
)
from yupi.tracking.undistorters import (
    Undistorter,
    ClassicUndistorter,
    RemapUndistorter,
    NoUndistorter
)

__all__ = [
    'ROI',
    'ObjectTracker',
    'CameraTracker',
    'TrackingScenario',
    'Algorithm',
    'IntensityMatching',
    'Undistorter',
    'ClassicUndistorter',
    'RemapUndistorter',
    'NoUndistorter'
]