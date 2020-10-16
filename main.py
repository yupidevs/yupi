import cv2
import os
import numpy as np
from tools import frame_diff
from settings import data_folder, data_file

# Initialize Video Source
cap = cv2.VideoCapture(os.path.join(os.getcwd(), data_folder, data_file))

# Temporal variables
previous_frame = None # Stores last processed frame

# Loop for all frames in the video
while True:
    # Get current frame
    ret, frame = cap.read()    
    if ret:
        # Checks if there is a real previous frame
        if not isinstance(previous_frame, np.ndarray):
            previous_frame = frame.copy()
            continue

        # Locates the region with higher pixel-change
        fd = frame_diff(previous_frame, frame)
        if isinstance(fd, np.ndarray):            
            cv2.imshow('a', fd)
            cv2.waitKey(10)
        else:
            break

        # Saves the current frame as 'previous' for next iteration
        previous_frame = frame.copy()   
        
    # Ends the processing when no more frames detected   
    else:
        print("[INFO] No more frames to process.")
        break

