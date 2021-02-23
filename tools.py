import os
import cv2
import json
import numpy as np
import settings as sett



def get_centroid(bin_img, hint=''):
    # Calculate moments
    M = cv2.moments(bin_img)  

    # Check if something was over the threshold
    if M['m00'] != 0:
        # calculate x,y coordinate of center
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        return cX, cY
    else:
        print('[ERROR] Nothing was over threshold\n {}'.format(hint))
        return 0, 0


def frame_diff_detector(frame1, frame2):
    # obtain the frame difference
    diff = cv2.absdiff(frame1, frame2)

    # convert image to grayscale image
    gray_image = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(gray_image, sett.frame_diff_threshold, 255, cv2.THRESH_BINARY)

    # give a hint message about what may go wrong
    hint = "Try decreasing the value of settings.frame_diff_threshold"
    
    # return the centroid of the pixels over threshold
    return get_centroid(thresh, hint)


def get_roi_bounds(dim, cXY):
    w, h = dim
    cX, cY = cXY
    # get the bounds of the roi
    xmin = max(cX - int(sett.roi_width/2), 0)
    xmax = min(cX + int(sett.roi_width/2), w)
    ymin = max(cY - int(sett.roi_heigh/2), 0)
    ymax = min(cY + int(sett.roi_heigh/2), h)
    return xmin, xmax, ymin, ymax


def get_roi(frame, cXY):
    h, w = frame.shape[:2]
    #bounds of the roi
    xmin, xmax, ymin, ymax = get_roi_bounds((w,h), cXY)
    window = frame[ymin:ymax, xmin:xmax, :]
    return window


def update_roi_center(img, prev_cXY):
    prev_cX, prev_cY = prev_cXY

    # get the centroid refered to the roi
    cX_roi, cY_roi = get_centroid(img)

    # get the centroid refered to the full image
    cX = prev_cX - int(sett.roi_width/2) + cX_roi
    cY = prev_cY - int(sett.roi_heigh/2) + cY_roi
    return cX, cY


def threshold_detector(frame):
    # convert image to grayscale image
    gray_image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

    # obtain image histogram
    ys = cv2.calcHist([gray_image], [0], None, [256], [0,256], True)

    # compute image total pixel count
    x, y = gray_image.shape
    total = x * y

    # compute an adaptative threshold according the ratio of darkest pixels
    min_threshold = sett.ant_darkest_pixel
    suma = 0
    for i in range(min_threshold, 256):
        suma += ys[i]
        if suma/total > sett.ant_ratio:
            max_threshold = i
            break

    # convert the grayscale image to binary image
    return cv2.inRange(gray_image, min_threshold, max_threshold)


def get_ant_mask(window):
    return threshold_detector(window)


def get_main_region(dim, border=sett.border):
    r = np.array(dim)
    dr = r * sett.border
    r1 = (1 - sett.border) * r / 2
    r2 = r1 + dr

    region_ = np.transpose([r1, r2])
    region = np.concatenate(region_).astype(np.int32)

    x0, xf, y0, yf = region.tolist()
    return x0, xf, y0, yf


def mask2track(dim, roi_array):
    w, h = dim
    mask = 255 * np.ones((h, w), dtype=np.uint8)

    # mask pixeles inside every ROIs
    for ROI in roi_array:
        x0, xf, y0, yf = get_roi_bounds(dim, ROI)
        mask[y0:yf, x0:xf] = 0

    return mask


def draw_frame(frame, cXY, region, features, frame_numb, mask):
    frame = frame.copy()
    h, w = frame.shape[:2]

    # draw region in which features are detected
    if region:
        x0, xf, y0, yf = region
        cv2.rectangle(frame, (x0, y0), (xf, yf), (0,0,255), 2)
    
    # draw detected and estimated features
    if features:
        p2, p3 = features
        for p2_, p3_ in zip(p2, p3):
            x2, y2 = np.rint(p2_).astype(np.int32)
            x3, y3 = np.rint(p3_).astype(np.int32)

            cv2.circle(frame, (x2,y2), 3, (0,0,0), -1)
            cv2.circle(frame, (x3,y3), 3, (0,255,0), -1)

    # draw a point over the ant and roi bounds
    if cXY:
        x1, x2, y1, y2 = get_roi_bounds((w, h), cXY)
        cv2.circle(frame, cXY, 5, (255, 255, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)

    # draw current frame number
    if frame_numb:
        x_, y_ = .02, .05
        x, y = int(x_ * w), int(y_ * h)
        cv2.putText(frame, str(frame_numb), (x, y), 
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, (0,255,255), 1, cv2.LINE_AA)

    # draw in black ant ROIs from previous and current frame
    if mask is not None:
        idx = np.where(mask == 0)
        frame[idx] = 0

    return frame


def resize_frame(frame, scale=sett.resize_factor):
    h, w = frame.shape[:2]
    w_, h_ = int(scale * w), int(scale * h)
    short_frame = cv2.resize(frame, (w_, h_), interpolation=cv2.INTER_AREA)
    return short_frame


def show_frame(frame, cXY=None, region=None, features=None, frame_numb=None,
        scale=sett.resize_factor, win_name=sett.video_file[:-4], mask=None):
    frame = draw_frame(frame, cXY, region, features, frame_numb, mask)
    frame = resize_frame(frame, scale)
    cv2.imshow(win_name, frame)
    return


def save_data(data, minutes=None, percent=None):
    data_file_name = 'data_' + sett.video_file[:-4]
    
    data_file_name += '_[' if (minutes or percent) else ''
    data_file_name += f'{minutes:.1f}min' if minutes else ''
    data_file_name += '-' if (minutes and percent) else ''
    data_file_name += f'{percent * 100 :.1f}%' if percent else ''
    data_file_name += ']' if (minutes or percent) else ''
    data_file_name += '.json'

    data_file_dir = os.path.join(sett.data_folder, data_file_name)
    with open(data_file_dir, 'w') as json_file:
        json.dump(data, json_file)
    
    return


def cXY_from_click(frame, 
                   win1_name='Clic the ant and press a key', 
                   win2_name='ROI'):
    
    frame_ = resize_frame(frame, scale=sett.resize_factor)
    cv2.imshow(win1_name, frame_)

    # callback handler to manually set the roi
    def on_click(event, x, y, flags, param):
        global cXY

        if event == cv2.EVENT_LBUTTONDOWN:
            # global roi center coordinates
            cXY = int(x / sett.resize_factor), int(y / sett.resize_factor)

            # copy of true frame and its resized version
            img = frame.copy()
            img_ = frame_.copy()

            # draw a circle in the selected pixel
            cv2.circle(img_, (x,y), 3, (0,255,255), 1)
            cv2.imshow(win1_name, img_)
            
            # get roi in the full size frame
            cv2.circle(img, cXY, 3, (0,255,255), 1)
            roi = get_roi(img, cXY)

            # roi padding just to display the new window
            padL, padR = np.hsplit(np.zeros_like(roi), 2)
            roi_ = np.hstack([padL, roi, padR])
            cv2.imshow(win2_name, roi_)

            print('ROI Initialized, now press any key to continue')
        return
    
    cv2.setMouseCallback(win1_name, on_click)
    cv2.waitKey(0)
    return cXY


class Undistorter():

    """Undistorts images using camera calibration matrix"""

    def __init__(self, method, camera_file):
        # Select the undistort method
        if method == "undistort":
            self.undistort = self.classic_undistort
        elif method == "remap":
            self.undistort = self.remap
        else:
            print('Undistort method not recognized. Undistort disabled')
            self.undistort = self.no_undistort

        # Read camera undistort matrix
        npzfile = os.path.join(os.getcwd(), 'cameras', camera_file)
        npzfile = np.load(npzfile)

        # Initialize camera parameters
        self.c_h = npzfile['h']
        self.c_w = npzfile['w']
        self.c_mtx = npzfile['mtx']
        self.c_dist = npzfile['dist']
        self.c_newcameramtx = npzfile['newcameramtx']
        self.c_mapx, self.c_mapy = cv2.initUndistortRectifyMap(self.c_mtx, self.c_dist, 
                                                               None, self.c_newcameramtx,
                                                               (self.c_w, self.c_h), 5)
        self.mask = None

    # Undistort by opencv undistort
    def classic_undistort(self, frame):
        return cv2.undistort(frame, self.c_mtx, self.c_dist, None, self.c_newcameramtx)
    
    # Undistort by opencv remapping
    def remap(self, frame):        
        return cv2.remap(frame, self.c_mapx, self.c_mapy, cv2.INTER_LINEAR)
    
    # No undistort
    def no_undistort(self, frame):
        return frame

    # method to be called to fix the distortion
    def fix(self, frame):
        if self.mask is None:
            self.create_mask(frame)
        corrected = self.undistort(frame)
        return self.masked(corrected)

    # create a mask with the distortion pattern
    def create_mask(self, frame):
        empty_frame = 255 * np.ones(frame.shape, dtype=np.uint8)        
        corrected = self.undistort(empty_frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        self.mask = cv2.erode(corrected, kernel, iterations=1)
        self.mask = cv2.bitwise_not(self.mask)
        self.background = np.full(frame.shape, 130, dtype=np.uint8)

    # apply the mask to a frame to adjust border colors
    def masked(self, frame):
        return cv2.bitwise_or(frame, self.mask)
