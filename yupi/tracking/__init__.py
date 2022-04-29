"""
This module is designed to capture Trajectory objects from
a sequence of images. It contains different tools to extract
the trajectories based on the color or the motion of the object
being tracked.

All the resources of this module should be imported directly
from ``yupi.tracking``.
"""

from yupi.tracking.algorithms import (
    BackgroundEstimator,
    BackgroundSubtraction,
    ColorMatching,
    FrameDifferencing,
    OpticalFlow,
    TemplateMatching,
    TrackingAlgorithm,
)
from yupi.tracking.trackers import ROI, CameraTracker, ObjectTracker, TrackingScenario
from yupi.tracking.undistorters import ClassicUndistorter, RemapUndistorter, Undistorter

__all__ = [
    "ROI",
    "ObjectTracker",
    "CameraTracker",
    "TrackingScenario",
    "TrackingAlgorithm",
    "FrameDifferencing",
    "ColorMatching",
    "BackgroundSubtraction",
    "TemplateMatching",
    "OpticalFlow",
    "BackgroundEstimator",
    "Undistorter",
    "ClassicUndistorter",
    "RemapUndistorter",
]
