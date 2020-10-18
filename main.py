import cv2
import os
import time
import json
import numpy as np
from tools import frame_diff_detector, get_ant_mask, update_roi_center, get_roi, Undistorter, show_frame, validate, get_possible_regions
from affine_estimator import get_affine
            
import settings as sett

# Initialize Video Source
cap = cv2.VideoCapture(os.path.join(os.getcwd(), sett.data_folder, sett.data_file))

# Temporal variables
cX, cY = None, None # center of the ROI
iteration = 0       # iteration counter
ant_coords = []
world_coords = [(0,0,0)]

# Initialize Spherical undistorter
U = Undistorter(sett.correction_method, sett.camera_correction_matrix)

# callback handler to manually set the roi
def on_click(event, x, y, p1, p2):
    global cX, cY
    if event == cv2.EVENT_LBUTTONDOWN:
        cX, cY = x, y
        print('ROI Initialized, now press any key to continue')


if __name__ == '__main__':
    # Loop for all frames in the video
    while True:

        # Skip some frames in the begining
        if iteration <= sett.skip_frames:
            ret, previous_frame = cap.read() 
            if sett.correct_spherical_distortion:
                previous_frame = U.fix(previous_frame) 
            
            print('Skipping frame {}'.format(iteration))
            iteration += 1
            continue

        # Get some regions to track the floor
        h, w, _ = previous_frame.shape
        regions = get_possible_regions(w, h)

        # Get current frame
        ret, frame = cap.read()    
        if ret:

            # Correct Spherical Distortion
            if sett.correct_spherical_distortion:
                frame = U.fix(frame) 

            # Initialize the center of the ROI 
            if not cX:

                # Init the ROI by frame differencing
                if sett.roi_initialization == 'auto':
                    cX, cY = frame_diff_detector(previous_frame, frame)

                # Init the ROI by user input
                else:
                    window_name = 'Clic the ant and press a key'
                    print('Clic over the ant in the image')
                    cv2.imshow(window_name, frame)
                    cv2.setMouseCallback(window_name, on_click)
                    cv2.waitKey(-1)

                    if not cX:
                        print("[ERROR] ROI was not Initialized")
                        break
                    else:
                        cv2.destroyAllWindows()                   

            # Get only the ROI from the current frame
            window = get_roi(frame, cX, cY)
            
            # Segmentate the ant inside the ROI
            ant_mask = get_ant_mask(window)

            # Alter the blue channel in ant-related pixels
            window[:,:,0] = ant_mask

            # update the roi center using current ant coordinates
            cX, cY = update_roi_center(ant_mask, cX, cY)

            # draw a point over the ant
            cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)


            # Track the floor
            valid_regions = validate(regions, cX, cY)
            tx, ty, angle, scale, region = get_affine(frame, previous_frame, valid_regions, show=False, debug=False)
            
            # save current frame as previous for next iteration
            previous_frame = frame

            # display the full image with the ant in blue
            show_frame(frame, cX, cY, region)

            # update data
            x_0, y_0, theta_0 = world_coords[-1]
            theta = theta_0 + angle
            x = tx * np.cos(theta) + ty * np.sin(theta) + x_0
            y = -tx * np.sin(theta) + ty * np.cos(theta) + y_0
            world_coords.append((x, y, theta))
            ant_coords.append((cX, cY))

        # Ends the processing when no more frames detected   
        else:
            print("[INFO] No more frames to process.")
            break

    cap.release()
    cv2.destroyAllWindows()

    data = {
        'ant_coords' : ant_coords,
        'camera_coords': world_coords
    }

    with open("data.json", "w") as write_file:
        json.dump(data, write_file)
