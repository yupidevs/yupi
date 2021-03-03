from tracking.trackers import ROI, ObjectTracker, CameraTracker, TrackingScenario
from tracking.undistorters import RemapUndistorter
from tracking.show_path import plot_results
from tracking.algorithms import IntensityMatching

if __name__ == '__main__':
    # Initialize main tracking objects
    algorithm = IntensityMatching(20, 180, 180)
    ant = ObjectTracker('ant', algorithm, ROI((120, 120)))
    camera = CameraTracker(ROI((.65, .65), ROI.CENTER_INIT_MODE))
    undistorter = RemapUndistorter('yupi/cameras/gph3+1080-60fps-NARROW.npz')
    scenario = TrackingScenario([ant], camera, undistorter)

    # Track the video using the preconfigured scenario
    retval, message = scenario.track('videos/video2_short.mp4')
    print(message)
    # plot_results('videos/video2_short_[0.2min-100.0%].json')