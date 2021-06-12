import abc
import numpy as np
import cv2


class Undistorter(metaclass=abc.ABCMeta):
    """
    Abstract class to model an undistortion method to be aplied on
    images in order to correct the spherical distortion caused by
    the camera lens. Classes inheriting from this class should
    implement ``undistort`` method.

    To use an undistorion method you will need to obtain the calibration
    matrix of your camera. You can follow the guide in opencv docs
    until the end of the section ``Undistortion`` to compute the matrix
    of your camera:

    https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

    Then you can save the matrix and other parameters in an npz file
    using numpy:

    >>> np.savez("camera_file.npz", h=h, w=w, mtx=mtx, dist=dist, newcameramtx=newcameramtx)

    Parameters
    ----------
    camera_file : str
        Path to the camera calibration file ("camera_file.npz" in the
        above example).
    """

    def __init__(self, camera_file):
        # Read camera undistort matrix
        npzfile = np.load(camera_file)

        # Initialize camera parameters
        self.c_h = npzfile['h']
        self.c_w = npzfile['w']
        self.c_mtx = npzfile['mtx']
        self.c_dist = npzfile['dist']
        self.c_newcameramtx = npzfile['newcameramtx']
        c_map = cv2.initUndistortRectifyMap(
            cameraMatrix=self.c_mtx,
            distCoeffs=self.c_dist,
            R=None,
            newCameraMatrix=self.c_newcameramtx,
            size=(self.c_w, self.c_h),
            m1type=5
        )
        self.c_mapx, self.c_mapy = c_map
        self.mask = None
        self.background = None

    @abc.abstractmethod
    def undistort(self, frame):
        """
        Abstract method that is implemented on inheriting classes. It
        should compute an undistorted version of frame using the given
        camera calibration matrix and a method specific to the
        inheriting class.
        """

    # Method to be called to fix the distortion
    def fix(self, frame):
        if self.mask is None:
            self.create_mask(frame)
        corrected = self.undistort(frame)
        return self.masked(corrected)

    # Create a mask with the distortion pattern
    def create_mask(self, frame):
        empty_frame = 255 * np.ones(frame.shape, dtype=np.uint8)
        corrected = self.undistort(empty_frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.mask = cv2.erode(corrected, kernel, iterations=1)
        self.mask = cv2.bitwise_not(self.mask)
        self.background = np.full(frame.shape, 130, dtype=np.uint8)

    # Apply the mask to a frame to adjust border colors
    def masked(self, frame):
        return cv2.bitwise_or(frame, self.mask)


class ClassicUndistorter(Undistorter):
    """
    Undistorter that performs undistortion using ``cv2.undistort``.

    Parameters
    ----------
    camera_file : str
        Path to the camera calibration file ("camera_file.npz" in the
        above example).
    """

    def undistort(self, frame):
        """
        Computes the undistorted version of ``frame`` using
        ``cv2.undistort``.

        Returns
        ----------
        np.ndarray
            Undistorted version of frame.
        """

        return cv2.undistort(
            src=frame,
            cameraMatrix=self.c_mtx,
            distCoeffs=self.c_dist,
            dst=None,
            newCameraMatrix=self.c_newcameramtx
        )


class RemapUndistorter(Undistorter):
    """
    Undistorter that performs undistortion using ``cv2.remap``.

    Parameters
    ----------
    camera_file : str
        Path to the camera calibration file ("camera_file.npz" in the
        above example).
    """

    def undistort(self, frame):
        """
        Computes the undistorted version of ``frame`` using
        ``cv2.remap``.

        Returns
        ----------
        np.ndarray
            Undistorted version of frame.
        """

        return cv2.remap(frame, self.c_mapx, self.c_mapy, cv2.INTER_LINEAR)
