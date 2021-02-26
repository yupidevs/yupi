from trackers import ROI, ObjectTracker, CameraTracker, TrackingScenario
from undistorters import RemapUndistorter
import tools

if __name__ == '__main__':
    # Initialize main tracking objects
    ant = ObjectTracker('ant', 'ColorMatching', ROI((120, 120), 'manual'))
    camera = CameraTracker(ROI((0.8, 0.8), 'center'))
    undistorter = RemapUndistorter('cameras/gph3+1080-60fps-NARROW.npz')
    scenario = TrackingScenario([ant], camera, undistorter)

    # Track the video using the preconfigured scenario
    retval, message = scenario.track('videos/video2_short.mp4')
    print(message)
