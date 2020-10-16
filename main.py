import cv2
import os
import numpy as np
from tools import frame_diff_detector, get_ant_mask, update_roi_center, get_roi
from settings import data_folder, data_file

# Initialize Video Source
cap = cv2.VideoCapture(os.path.join(os.getcwd(), data_folder, data_file))

# Temporal variables
previous_frame = None # Stores last processed frame
first_time = True

if __name__ == '__main__':    
    # Loop for all frames in the video
    while True:
        # Get current frame
        ret, frame = cap.read()    
        if ret:

            # Checks if there is a real previous frame
            if not isinstance(previous_frame, np.ndarray):
                previous_frame = frame.copy()
                continue

            # Initialize the center of the ROI by frame differencing
            if first_time:
                cX, cY = frame_diff_detector(previous_frame, frame)
                first_time = False


            # Get only the ROI from the current frame
            window = get_roi(frame, cX, cY)
            
            # Segmentate the ant inside the ROI
            ant_mask = get_ant_mask(window)

            # Alter the blue channel in ant-related pixels
            window[:,:,0] = ant_mask

            # update the roi center using current ant coordinates
            cX, cY = update_roi_center(ant_mask, cX, cY)

            # display the full image with the ant in blue
            cv2.imshow('a', frame)
            cv2.waitKey(10)
            

        # Ends the processing when no more frames detected   
        else:
            print("[INFO] No more frames to process.")
            break

    cap.release()
    cv2.destroyAllWindows()