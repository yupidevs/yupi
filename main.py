import cv2
import os
import numpy as np
from tools import frame_diff_detector, get_ant_mask, update_roi_center, get_roi, Undistorter
            
import settings as sett

# Initialize Video Source
cap = cv2.VideoCapture(os.path.join(os.getcwd(), sett.data_folder, sett.data_file))

# Temporal variables
cX, cY = None, None
iteration = 0    

# Initialize Spherical undistorter
U = Undistorter(sett.correction_method, sett.camera_correction_matrix)

if __name__ == '__main__':
    # Loop for all frames in the video
    while True:

        # Skip some frames in the begining
        if iteration <= sett.skip_frames:
            ret, previous_frame = cap.read() 
            if sett.correct_spherical_distortion:
                previous_frame = U.undistort(previous_frame) 
            print('Skipping frame {}'.format(iteration))
            iteration += 1
            continue          

        # Get current frame
        ret, frame = cap.read()    
        if ret:

            # Correct Spherical Distortion
            if sett.correct_spherical_distortion:
                frame = U.undistort(frame) 

            # Initialize the center of the ROI by frame differencing
            if not cX:
                cX, cY = frame_diff_detector(previous_frame, frame)

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