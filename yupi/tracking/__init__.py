"""
This submodule is designed to capture Trajectory objects from
a sequence of images. It contains different tools to extract
the trajectories based on the color or the motion of the object
being tracked.

All the resources of this module should be imported directly
from ``yupi.tracking``. However, we ordered the resources
according the functionality into the following categories.
"""

from yupi.tracking.trackers import (
    ROI,
    ObjectTracker,
    CameraTracker,
    TrackingScenario
)
from yupi.tracking.algorithms import (
    TrackingAlgorithm,
    IntensityMatching,
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
    'IntensityMatching',
    'FrameDifferencing',
    'ColorMatching',
    'BackgroundSubtraction',
    'TemplateMatching',
    'OpticalFlow',
    'BackgroundEstimator',
    'Undistorter',
    'ClassicUndistorter',
    'RemapUndistorter',
    'NoUndistorter'
]
