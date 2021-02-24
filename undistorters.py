import os
import cv2
import numpy as np
import abc

class Undistorter(metaclass=abc.ABCMeta):

    """Undistorts images using camera calibration matrix"""

    def __init__(self, camera_file):

        # Read camera undistort matrix
        npzfile = np.load(camera_file)

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

    @abc.abstractmethod
    def undistort(self, frame):
        pass

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


class ClassicUndistorter(Undistorter):
    """docstring for ClassicUndistorter"""
    def __init__(self, camera_file):
        super(ClassicUndistorter, self).__init__(camera_file)

    def undistort(self, frame):
        return cv2.undistort(frame, self.c_mtx, self.c_dist, None, self.c_newcameramtx)  


class RemapUndistorter(Undistorter):
    """docstring for RemapUndistorter"""
    def __init__(self, camera_file):
        super(RemapUndistorter, self).__init__(camera_file)

    def undistort(self, frame):
        return cv2.remap(frame, self.c_mapx, self.c_mapy, cv2.INTER_LINEAR)


class NoUndistorter(Undistorter):
    """docstring for NoUndistorter"""
    def __init__(self, camera_file):
        super(NoUndistorter, self).__init__(camera_file)

    def undistort(self, frame):
        return frame
