import cv2
import numpy as np
import os
from settings import frame_diff_threshold, roi_width, roi_heigh, ant_ratio


def get_centroid(bin_img, hint=''):
    # Calculate moments
    M = cv2.moments(bin_img)  

    # Checks if something was over the threshold
    if M["m00"] != 0:
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    else:
        print('[ERROR] Nothing was over threshold\n {}'.format(hint))
        return 0, 0

def update_roi_center(img, cX_old, cY_old):
    # get the centroid refered to the roi
    cX_roi, cY_roi = get_centroid(img)

    # get the centroid refered to the full image
    cX = cX_old - int(roi_width/2) + cX_roi
    cY = cY_old - int(roi_heigh/2) + cY_roi
    return cX, cY


def get_roi(frame, cX, cY):
    # get image width and height
    h, w, _ = frame.shape

    # get the bounds of the roi
    ymin = max(cY - int(roi_heigh/2), 0)
    ymax = min(cY + int(roi_heigh/2), h)
    xmin = max(cX-int(roi_width/2), 0)
    xmax = min(cX+int(roi_width/2), w)
    return frame[ymin:ymax, xmin:xmax, :]


def frame_diff_detector(frame1, frame2):
    # obtain the frame difference
    diff = cv2.absdiff(frame1, frame2)

    # convert image to grayscale image
    gray_image = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(gray_image, frame_diff_threshold, 255, cv2.THRESH_BINARY)

    # give a hint message about what may go wrong
    hint = "Try decreasing the value of settings.frame_diff_threshold"
    
    # return the centroid of the pixels over threshold
    return get_centroid(thresh, hint)


def threshold_detector(frame):
    # convert image to grayscale image
    gray_image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

    # obtain image histogram
    ys = cv2.calcHist([gray_image], [0], None, [256], [0,256], True)

    # compute image total pixel count
    x, y = gray_image.shape
    total = x * y

    # compute an adaptative threshold according the ratio of darkest pixels
    suma = 0
    for i in range(3, 256):
        suma += ys[i]
        if suma/total > ant_ratio:
            threshold = i
            break

    # convert the grayscale image to binary image
    return cv2.inRange(gray_image, 3, threshold)


def get_ant_mask(window):
    return threshold_detector(window)

def show_frame(frame, scale=0.5):
    h, w, _ = frame.shape
    short_frame = cv2.resize(frame, (int(scale * w), int(scale * h)), interpolation = cv2.INTER_AREA)
    cv2.imshow('Current Frame', short_frame)
    cv2.waitKey(10)


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
        self.c_h = npzfile["h"]
        self.c_w = npzfile["w"]
        self.c_mtx = npzfile["mtx"]
        self.c_dist = npzfile["dist"]
        self.c_newcameramtx = npzfile["newcameramtx"]
        self.c_mapx, self.c_mapy = cv2.initUndistortRectifyMap(self.c_mtx, self.c_dist, None, self.c_newcameramtx,(self.c_w, self.c_h),5)
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
        self.mask = cv2.erode(corrected, kernel, iterations = 1)
        self.mask = cv2.bitwise_not(self.mask)
        self.background = np.full(frame.shape, 130, dtype=np.uint8)

    # apply the mask to a frame to adjust border colors
    def masked(self, frame):
        return cv2.bitwise_or(frame, self.mask)
