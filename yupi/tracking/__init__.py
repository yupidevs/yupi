"""
This module is designed to capture Trajectory objects from
a sequence of images. It contains different tools to extract
the trajectories based on the color or the motion of the object
being tracked.

All the resources of this module should be imported directly
from ``yupi.tracking``.
"""

from yupi.tracking.trackers import (
    ROI,
    ObjectTracker,
    CameraTracker,
    TrackingScenario
)
from yupi.tracking.algorithms import (
    TrackingAlgorithm,
    ColorMatching,
    FrameDifferencing,
    BackgroundSubtraction,
    TemplateMatching,
    OpticalFlow,
    BackgroundEstimator
)
from yupi.tracking.undistorters import (
    Undistorter,
    ClassicUndistorter,
    RemapUndistorter
)

__all__ = [
    'ROI',
    'ObjectTracker',
    'CameraTracker',
    'TrackingScenario',
    'TrackingAlgorithm',
    'FrameDifferencing',
    'ColorMatching',
    'BackgroundSubtraction',
    'TemplateMatching',
    'OpticalFlow',
    'BackgroundEstimator',
    'Undistorter',
    'ClassicUndistorter',
    'RemapUndistorter',
]
