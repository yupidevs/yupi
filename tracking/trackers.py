import os
import cv2
import json
import tracking.tools
import numpy as np
import tracking.settings
from tracking.affine_estimator import get_affine

import tracking.tools as tools

class ROI():
    """docstring for ROI"""
    def __init__(self, roi_size, init_method, scale=0.5):
        self.roi_width, self.roi_heigh = roi_size
        self.prev_cXY = None, None
        self.cXY = None, None
        self.roi_init_mode = init_method
        self.scale = scale

    def get_bounds(self):
        cX, cY = self.cXY
        # get the bounds of the roi
        xmin = max(cX - int(self.roi_width/2), 0)
        xmax = min(cX + int(self.roi_width/2), self.global_width)
        ymin = max(cY - int(self.roi_heigh/2), 0)
        ymax = min(cY + int(self.roi_heigh/2), self.global_heigh)
        return xmin, xmax, ymin, ymax

    def center_init(self, frame, name):
        self.global_heigh, self.global_width = frame.shape[:2]
        self.cXY = int(self.global_width/2), int(self.global_heigh/2)
        return self.cXY

    def manual_init(self, frame, name,
                       win2_name='ROI'):

        win1_name = 'Clic on the center of {} to init roi'.format(name)

        self.global_heigh, self.global_width = frame.shape[:2]

        frame_ = tools.resize_frame(frame, scale=self.scale)
        cv2.imshow(win1_name, frame_)

        # callback handler to manually set the roi
        def on_click(event, x, y, flags, param):

            if event == cv2.EVENT_LBUTTONDOWN:
                # global roi center coordinates
                self.cXY = int(x / self.scale), int(y / self.scale)

                # copy of true frame and its resized version
                img = frame.copy()
                img_ = frame_.copy()

                # draw a circle in the selected pixel
                cv2.circle(img_, (x,y), 3, (0,255,255), 1)
                cv2.imshow(win1_name, img_)
                
                # get roi in the full size frame
                cv2.circle(img, self.cXY, 3, (0,255,255), 1)
                roi = self.crop(img)

                # roi padding just to display the new window
                padL, padR = np.hsplit(np.zeros_like(roi), 2)
                roi_ = np.hstack([padL, roi, padR])
                cv2.imshow(win2_name, roi_)

                print('ROI Initialized, now press any key to continue')
        
        cv2.setMouseCallback(win1_name, on_click)
        cv2.waitKey(0)
        return self.cXY

    def __check_roi_init__(self, name):
        if not self.prev_cXY[0]:
            return False, '[ERROR] ROI was not Initialized (in {})'.format(name)
        else:
            cv2.destroyAllWindows()
            return True, '[INFO] ROI was Initialized (in {})'.format(name)

    def initialize(self, name, first_frame):
        h, w = first_frame.shape[:2]
        if self.roi_width <= 1:
            self.roi_width *= w
        if self.roi_heigh <= 1:
            self.roi_heigh *= h

        # Initialize ROI coordinates manually by user input
        if self.roi_init_mode == 'manual':
            self.cXY = self.manual_init(first_frame, name)
            self.prev_cXY = self.cXY
        elif self.roi_init_mode == 'center':
            self.cXY = self.center_init(first_frame, name)
            self.prev_cXY = self.cXY
        else:
            return False, '[ERROR] ROI initialization mode unknown (in {})'.format(name)
        return self.__check_roi_init__(name)

    def crop(self, frame):
        self.global_heigh, self.global_width = frame.shape[:2]
        #bounds of the roi
        xmin, xmax, ymin, ymax = self.get_bounds()
        window = frame[ymin:ymax, xmin:xmax, :]
        return window

    def update_center(self):
        pass

class ObjectTracker():
    """docstring for ObjectTracker"""
    def __init__(self, name, method, roi):
        self.name = name
        self.roi = roi
        self.history = []

    def __init_roi__(self, prev_frame):
        return self.roi.initialize(self.name, prev_frame)

    def track(self, frame):
        # get only the ROI from the current frame
        window = self.roi.crop(frame)
        
        # segmentate the ant inside the ROI
        ant_mask = tools.get_ant_mask(window)

        # alter the blue channel in ant-related pixels
        window[:,:,0] = ant_mask

        # update the roi center using current ant coordinates
        self.roi.cXY = tools.update_roi_center(ant_mask, self.roi.prev_cXY)

        # update data
        self.history.append(self.roi.cXY)
        self.roi.prev_cXY = self.roi.cXY
  


class CameraTracker():
    """docstring for CameraTracker"""
    def __init__(self, roi):
        self.history = []
        self.mse = []
        self.roi = roi

    def __init_roi__(self, prev_frame):
        return self.roi.initialize('Camera', prev_frame)

    # track the floor
    def track(self, prev_frame, frame, ignored_regions):
        # Initialize a mask of what to track
        h, w = frame.shape[:2]
        mask = 255 * np.ones((h, w), dtype=np.uint8)

        # mask pixeles inside every ROIs
        for x0, xf, y0, yf in ignored_regions:
            mask[y0:yf, x0:xf] = 0

        p_good, aff_params, err = get_affine(prev_frame, frame, self.roi.get_bounds(), mask)
        self.features = p_good[1:]

        if err is None:
            return False, 'No matrix was estimated'

        self.history.append(aff_params)
        self.mse.append(err)

        return True, 'Camera Tracked'
        

class TrackingScenario():
    """docstring for TrackingScenario"""
    def __init__(self, object_trackers, camera_tracker=None, undistorter=None):
        self.object_trackers = object_trackers   
        self.camera_tracker = camera_tracker 
        self.iteration_counter = 0
        self.auto_mode = True
        self.undistorter = undistorter
        self.enabled = True

    def __digest_video_path(self, video_path):
        # TODO: Validate the path
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)                        # create capture object
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) # total number of frames in the video file
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)                      # frames per seconds
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # frame width
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          # frame height
        self.dim = (self.w, self.h)

        self.first_frame = 0

    def __undistort__(self, frame):
        if self.undistorter:
            frame = self.undistorter.fix(frame)
        return frame


    def show_frame(self, frame, show_frame_id=True):
        # cXY, region, features, frame_numb, mask
        frame = frame.copy()

        # draw region in which features are detected
        if self.camera_tracker:
            x0, xf, y0, yf = self.camera_tracker.roi.get_bounds()
            cv2.putText(frame, 'Camera Tracking region', (x0+5, yf-5), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0,0,255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x0, y0), (xf, yf), (0,0,255), 2)
            p2, p3 = self.camera_tracker.features
            # draw detected and estimated features
            for p2_, p3_ in zip(p2, p3):
                x2, y2 = np.rint(p2_).astype(np.int32)
                x3, y3 = np.rint(p3_).astype(np.int32)

                cv2.circle(frame, (x2,y2), 3, (0,0,0), -1)
                cv2.circle(frame, (x3,y3), 3, (0,255,0), -1)

        for otrack in self.object_trackers:
            # draw a point over the roi center and draw bounds
            x1, x2, y1, y2 = otrack.roi.get_bounds()
            cv2.circle(frame, otrack.roi.cXY, 5, (255, 255, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
            cv2.putText(frame, otrack.name, (x1+5, y2-5), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0,255,255), 1, cv2.LINE_AA)

        if show_frame_id:
            h, w = frame.shape[:2]
            frame_id = self.iteration_counter + self.first_frame
            x_, y_ = .02, .05
            x, y = int(x_ * w), int(y_ * h)
            cv2.putText(frame, str(frame_id), (x, y), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0,255,255), 1, cv2.LINE_AA)

        frame = tools.resize_frame(frame)
        cv2.imshow('PYTANAL processing window', frame)
        # return frame

    def __first_iteration__(self, start_in_frame):
        # Start processing frams at the given index
        if start_in_frame:
            self.first_frame = start_in_frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_in_frame)

        # Capture the first frame to process
        ret, prev_frame = self.cap.read()
        
        # correct spherical distortion
        self.prev_frame = self.__undistort__(prev_frame)

        # Initialize the roi of all the trackers
        for otrack in self.object_trackers:
            retval, message = otrack.__init_roi__(self.prev_frame)
            if not retval:
                return retval, message

        # Initialize the region of the camera tracker
        if self.camera_tracker:
            self.camera_tracker.__init_roi__(self.prev_frame)

        # Increase the iteration counter
        self.iteration_counter += 1

        return True, '[INFO] All trackers were initialized'

    def keyboard_controller(self):
        # keyboard events
        wait_key = 0 if not self.auto_mode else 10

        k = cv2.waitKey(wait_key) & 0xff
        if k == ord('m'):
            self.auto_mode = not self.auto_mode

        if k == ord('s'):
            self.__export_data__()

        elif k == ord('q'):
            self.enabled = False

        elif k == ord('e'):
            exit()

    def __regular_iteration__(self):
        # get current frame and ends the processing when no more frames are detected
        ret, frame = self.cap.read()
        if not ret: 
            return False, '[INFO] All frames were processed.'
        
        # correct spherical distortion
        frame = self.__undistort__(frame)

        # ROI Arrays of tracking objects
        roi_array = []

        # Track every object and save past and current ROIs
        for otrack in self.object_trackers:
            roi_array.append(otrack.roi.get_bounds())
            otrack.track(frame)
            roi_array.append(otrack.roi.get_bounds())
 
        
        ret, message = self.camera_tracker.track(self.prev_frame, frame, roi_array)
        frame_id = self.iteration_counter + self.first_frame

        if not ret:
            return False, '[Error] {} (Frame {})'.format(message, frame_id)
       

        # display the full image with the ant in blue (TODO: Refactor this to make more general)
        self.show_frame(frame)

        # save current frame and ROI center as previous for next iteration
        self.prev_frame = frame.copy()


        # Call the keyboard controller to handle key interruptions
        self.keyboard_controller()

        # Increase the iteration counter
        self.iteration_counter += 1

        return True, '[INFO] Frame {} was processed'.format(frame_id)

    def __release_cap__(self):
        self.cap.release()
        cv2.destroyAllWindows()


    def __export_data__(self):
        # TODO: This function needs to be rewritten to be able to handle more than 1 object tracker
        # and also to provide more general purpose semantic information
        last_frame = self.first_frame + self.iteration_counter
        percent_video = 100 * (self.first_frame + self.iteration_counter) / self.frame_count
        minutes_video = last_frame / self.fps / 60

        data = {
            'fps': self.fps,
            'first_frame': self.first_frame,
            'last_frame': last_frame,
            'percent': percent_video,
            'r_ac' : self.object_trackers[0].history,
            'affine_params': self.camera_tracker.history,
            'mse': self.camera_tracker.mse
        }
        self.__save_data__(data, minutes_video, percent_video)


    def __save_data__(self, data, minutes=None, percent=None):
        if not (minutes or percent):
            progress = ''
        else:
            summary = f'{minutes:.1f}min' if minutes else ''
            summary += '-' if (minutes and percent) else ''
            summary += f'{percent:.1f}%' if percent else ''
            progress = '_[{}]'.format(summary)

        data_file_dir = '{}{}.json'.format(self.video_path[:-4], progress)

        with open(data_file_dir, 'w') as json_file:
            json.dump(data, json_file)

    def track(self, video_path, start_in_frame=0):
        self.__digest_video_path(video_path)
        
        if self.iteration_counter == 0:
            retval, message = self.__first_iteration__(start_in_frame)
            if not retval:
                return retval, message

        while self.enabled:
            retval, message = self.__regular_iteration__()
            if not retval:
                break

        if message == '[INFO] All frames were processed.':
            retval = True

        self.__release_cap__()
        self.__export_data__()
        return retval, message

