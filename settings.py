"""
Data-related stuff
"""
data_folder = 'data' # Folder containing exported data files
video_folder = 'videos' # Folder containing the videos
video_file = 'video2_short.mp4' # Video file being analized
first_frame = 1 # Frames to skip before to start processing
resize_factor = .5 # Factor to resize displayed frames


"""
Algorithm's parameters
"""
roi_width = 120 # Width of the region of image analized on each frame
roi_heigh = 120 # Heigh of the region of image analized on each frame
ant_pixels = 180 # Approximate area of the ant in pixels

ant_darkest_pixel = 20 # Darkest pixel that actually belongs to the ant

roi_initialization = 'manual' # Only 2 methods available: 'manual' or 'auto'
frame_diff_threshold = 15 # Only usefull when roi_initialization is in 'auto' mode
border = .65 # Fraction of the image used in features detection ((1 - border) is the padding)


"""
Camera-related stuff
"""
correct_spherical_distortion = True # Set to True only when decent calibration is available
correction_method = 'remap' # Only 2 methods available: 'remap' or 'undistort'
camera_correction_matrix = 'gph3+1080-60fps-NARROW.npz' # Available cameras are in 'cameras' folder


"""
Algorithm's constants
"""
ant_ratio = ant_pixels / (roi_width * roi_heigh) # approximate ratio of the ant compare to the roi
