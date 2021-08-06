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
    turn : bool
        This parameter is used to rotate 90 degrees the frame, before
        undistorting it. It is useful when the input video is rotated
        respect the orginal orientation used when the camera was
        calibrated (Not a very frequent use case). The undistorted
        result will be rotated -90 degrees before returning. By default
        is False.
    """

    def __init__(self, camera_file, turn=False):
        # Read camera undistort matrix
        npzfile = np.load(camera_file)

        # Initialize camera parameters
        self.c_h = npzfile['h']
        self.c_w = npzfile['w']
        self.c_mtx = npzfile['mtx']
        self.c_dist = npzfile['dist']
        self.c_newcameramtx = npzfile['newcameramtx']
        self.turn = turn
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

    # Turn the image if required
    def _rotate(self, frame, _input=True):
        if self.turn:
            direction = cv2.ROTATE_90_COUNTERCLOCKWISE
            if _input:
                direction = cv2.ROTATE_90_CLOCKWISE
            frame = cv2.rotate(frame, direction)
        return frame

    # Method to be called to fix the distortion
    def fix(self, frame):
        frame = self._rotate(frame, _input=True)
        if self.mask is None:
            self._create_mask(frame)
        corrected = self.undistort(frame)
        return self.masked(corrected)

    # Create a mask with the distortion pattern
    def _create_mask(self, frame):
        empty_frame = 255 * np.ones(frame.shape, dtype=np.uint8)
        corrected = self.undistort(empty_frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.mask = cv2.erode(corrected, kernel, iterations=1)
        self.mask = cv2.bitwise_not(self.mask)
        self.background = np.full(frame.shape, 130, dtype=np.uint8)

    # Apply the mask to a frame to adjust border colors
    def masked(self, frame):
        frame = cv2.bitwise_or(frame, self.mask)
        return  self._rotate(frame, _input=False)


class ClassicUndistorter(Undistorter):
    """
    Undistorter that performs undistortion using ``cv2.undistort``.

    Parameters
    ----------
    camera_file : str
        Path to the camera calibration file ("camera_file.npz" in the
        above example).
    turn : bool
        This parameter is used to rotate 90 degrees the frame, before
        undistorting it. It is useful when the input video is rotated
        respect the orginal orientation used when the camera was
        calibrated (Not a very frequent use case). The undistorted
        result will be rotated -90 degrees before returning. By default
        is False.
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
    turn : bool
        This parameter is used to rotate 90 degrees the frame, before
        undistorting it. It is useful when the input video is rotated
        respect the orginal orientation used when the camera was
        calibrated (Not a very frequent use case). The undistorted
        result will be rotated -90 degrees before returning. By default
        is False.
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
