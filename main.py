from trackers import ROI, ObjectTracker, CameraTracker, TrackingScenario
from undistorters import RemapUndistorter
import tools

if __name__ == '__main__':
    ant_roi = ROI((120, 120), 'manual')
    # Initialize main tracking objects
    ant = ObjectTracker('ant', 'ColorMatching', ant_roi)
    camera = CameraTracker()
    undistorter = RemapUndistorter('cameras/gph3+1080-60fps-NARROW.npz')
    scenario = TrackingScenario([ant], camera, undistorter)

    # Track the video using the preconfigured scenario
    retval, message = scenario.track('videos/video2_short.mp4')
    print(message)
