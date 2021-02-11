import os
import cv2
import tools
import numpy as np
import settings as sett
from affine_estimator import get_affine



# initialize video source
cap = cv2.VideoCapture(os.path.join(os.getcwd(), sett.video_folder, sett.video_file))

# variables
total_frame_numb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # total number of frames in the video file
fps = cap.get(cv2.CAP_PROP_FPS) # frames per seconds

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # frame width
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # frame height
dim = (w, h)

region = tools.get_main_region(dim) # main region in which to analyze features

cXY = None, None    # center of the ROI
iteration = 0       # iteration counter
manual_mode = False # passing frames by hand

r_ac = []          # ant position in the camera frame of reference
affine_params = [] # affine matrix parameters
mse = []           # mean square errors

# initialize spherical undistorter
U = tools.Undistorter(sett.correction_method, sett.camera_correction_matrix)


if __name__ == '__main__':
    # loop for all frames in the video
    while True:

        if iteration == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, sett.first_frame)
            ret, prev_frame = cap.read()
            
            # correct spherical distortion
            if sett.correct_spherical_distortion:
                prev_frame = U.fix(prev_frame)

            # init ROI coordinates manually by user input
            prev_cXY = (None, None)
            if sett.roi_initialization == 'manual':
                prev_cXY = tools.cXY_from_click(prev_frame)
                if not prev_cXY[0]:
                    print('[ERROR] ROI was not Initialized')
                    break
                else:
                    cv2.destroyAllWindows()
            
            iteration += 1
            continue

        # get current frame and ends the processing when no more frames are detected
        ret, frame = cap.read()
        if not ret: 
            print('[INFO] No more frames to process.')
            break
        
        # current frame number
        frame_numb = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        percent_video = frame_numb / total_frame_numb

        # correct spherical distortion
        if sett.correct_spherical_distortion:
            frame = U.fix(frame)

        # init ROI center automatically by frame differencing
        if sett.roi_initialization == 'auto':
            cXY = tools.frame_diff_detector(prev_frame, frame)
            prev_cXY = cXY
             
        # get only the ROI from the current frame
        window = tools.get_roi(frame, prev_cXY)
        
        # segmentate the ant inside the ROI
        ant_mask = tools.get_ant_mask(window)

        # alter the blue channel in ant-related pixels
        window[:,:,0] = ant_mask

        # update the roi center using current ant coordinates
        cXY = tools.update_roi_center(ant_mask, prev_cXY)

        # track the floor
        mask = tools.mask2track(dim, prev_cXY, cXY)
        p_good, aff_params, err = get_affine(prev_frame, frame, region, mask)
        features = p_good[1:]

        if err is None:
            print('No matrix was estimated in frame %i' % frame_numb)
            break

        # update data
        r_ac.append(cXY)
        affine_params.append(aff_params)
        mse.append(err)
        
        # display the full image with the ant in blue
        tools.show_frame(frame, cXY, region, features, frame_numb)

        # save current frame and ROI center as previous for next iteration
        prev_frame = frame.copy()
        prev_cXY = cXY

        # keyboard events
        wait_key = 0 if manual_mode else 10
        k = cv2.waitKey(wait_key) & 0xff

        if k == ord('m'):
            manual_mode = not manual_mode

        if k == ord('s'):
            data = {
                'fps': fps,
                'first_frame': sett.first_frame,
                'last_frame': frame_numb,
                'percent': percent_video,
                'r_ac' : r_ac,
                'affine_params': affine_params,
                'mse': mse
            }
            tools.save_data(data)

        elif k == ord('q'):
            break

        elif k == ord('e'):
            exit()
        
        iteration += 1

    cap.release()
    cv2.destroyAllWindows()
    
    
    data = {
        'fps': fps,
        'first_frame': sett.first_frame,
        'last_frame': frame_numb,
        'percent': percent_video,
        'r_ac' : r_ac,
        'affine_params': affine_params,
        'mse': mse
    }
    minutes_video = frame_numb / fps / 60
    tools.save_data(data, minutes_video, percent_video)
