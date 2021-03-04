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
    IntensityMatching,
    ColorMatching
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
    'ColorMatching',
    'Undistorter',
    'ClassicUndistorter',
    'RemapUndistorter',
    'NoUndistorter'
]