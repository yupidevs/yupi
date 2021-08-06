import abc
import logging
import cv2
import numpy as np


def _resize_frame(frame, scale=1):
    h, w = frame.shape[:2]
    w_, h_ = int(scale * w), int(scale * h)
    short_frame = cv2.resize(frame, (w_, h_), interpolation=cv2.INTER_AREA)
    return short_frame


def _change_colorspace(image, color_space):
    if color_space == 'BGR':
        return image
    if color_space == 'GRAY':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if color_space == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


class BackgroundEstimator():
    """
    This class provides static methods to determine the background in image
    sequences. It estimates the temporal median of the sequence.
    """

    def __init__(self):
        pass

    @staticmethod
    def from_video(video_path, samples, start_in=0):
        """
        This method takes a video indicated by ``video_path`` and
        uniformely take a number of image samples according to the
        parameter ``samples``. Then, it computes the temporal median of
        the images in order to determine de background.

        Parameters
        ----------
        video_path : str
            Path to the video file
        samples : int
            Number of samples to get from the video.
        start_in : int, optional
            If passed, the method will start sampling after the frame
            indicated by this value, by default 0.
        """

        # Create a cv2 Video Capture Object
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        effective_frames = total_frames - start_in
        spacing = effective_frames / samples

         # Store frames in a list
        frames = []
        for i in range(samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * spacing + start_in)
            _, frame = cap.read()
            frames.append(frame)

        # Calculate the median along time
        return np.median(frames, axis=0).astype(dtype=np.uint8)

class TrackingAlgorithm(metaclass=abc.ABCMeta):
    """
    Abstract class to model a Tracking Algorithm. Classes inheriting
    from this class should implement ``detect`` method.
    """

    def __init__(self):
        pass

    def get_centroid(self, bin_img):
        """
        Computes the centroid of a binary image using ``cv2.moments``.

        Parameters
        ----------
        bin_img : np.ndarray
            Binary image used to compute a centroid

        Returns
        -------
        tuple
            x, y coordinates of the centroid
        """

        # Calculate moments
        M = cv2.moments(bin_img)

        # Check if something was over the threshold
        if M['m00'] != 0:
            # Calculate x,y coordinate of center
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            return cX, cY
        logging.warning("Nothing was over threshold. Algorithm: %s",
                        type(self).__name__)
        return None

    def preprocess(self, frame, roi_bound, preprocessing):
        frame = frame.copy()
        if roi_bound:
            xmin, xmax, ymin, ymax = roi_bound
            frame = frame[ymin:ymax, xmin:xmax, :]
        if preprocessing is not None:
            frame = preprocessing(frame)
        return frame

    @abc.abstractmethod
    def detect(self, frame, roi_bound=None, preprocessing=None):
        """
        Abstract method that is implemented on inheriting classes.
        It should compute the location (in the image ``frame``)
        of the object being tracked.

        Parameters
        ----------
        frame : np.ndarray
            Image where the algorithm must identify the object
        roi_bound : tuple, optional
            Coordinates of the region of interest of the frame. The
            expected format if a tuple with the form (xmin, xmax, ymin,
            ymax). If passed the algorithm will crop this region of the
            frame and will proceed only in this region, providing the
            estimations refered to this region instead of the whole
            image, by default None.
        preprocessing : func
            A function to be applied to the frame (Or cropped version
            of it if roi_bound is passed) before detecting the object
            on it, by default None.
        """


class ColorMatching(TrackingAlgorithm):
    """
    Identifies the position of an object by thresholding pixel
    color values in the pre-defined ranges.

    Parameters
    ----------
    lower_bound : tuple, optional
        Minimum value of pixel color to be considered as part of
        the object, by default (0,0,0)
    upper_bound : tuple, optional
        Maximum value of pixel color to be considered as part of
        the object, by default (255,255,255)
    color_space : str, optional
        Color space to be used before thresholding with the given
        bounds. The image will be automatically converted to this
        color space, by default 'BGR'.
    max_pixels : int, optional
        If this parameter is passed, the algoritm will stop searching
        for candidate pixels after reaching a count equal to this value,
        by default None.
    """

    def __init__(self, lower_bound=(0, 0, 0), upper_bound=(255, 255, 255),
                 color_space='BGR', max_pixels=None):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.color_space = color_space
        self.max_pixels = max_pixels

    def detect(self, frame, roi_bound=None, preprocessing=None):
        """
        Identifies the tracked object in the image ``frame``
        by thresholding it using the bound parameters defined when
        the object was constructed.

        Parameters
        ----------
        frame : np.ndarray
            Image containing the object to be tracked
        roi_bound : tuple, optional
            Coordinates of the region of interest of the frame. The
            expected format if a tuple with the form (xmin, xmax, ymin,
            ymax). If passed the algorithm will crop this region of the
            frame and will proceed only in this region, providing the
            estimations refered to this region instead of the whole
            image, by default None.
        preprocessing : func
            A function to be applied to the frame (Or cropped version
            of it if roi_bound is passed) before detecting the object
            on it, by default None.

        Returns
        -------
        np.ndarray
            A binary version of ``frame`` where elements with value
            ``0`` indicate the absence of object and ``1`` the precense
            of the object.
        tuple
            x, y coordinates of the centroid of the object in the image.
        """

        # Make a preprocessed (and copied) version of the frame
        frame = self.preprocess(frame, roi_bound, preprocessing)

        # Convert image to desired colorspace
        copied_image = _change_colorspace(frame, self.color_space)

        mask = cv2.inRange(copied_image, self.lower_bound, self.upper_bound)
        centroid = self.get_centroid(mask)
        # Convert the grayscale image to binary image
        return mask, centroid


class FrameDifferencing(TrackingAlgorithm):
    """
    Identifies the position of an object by comparison between
    consecutive frames

    Parameters
    ----------
        Minimum difference (in terms of pixel intensity) among current
        and previous image to consider that pixel as part of a moving
        object, by default 1.
    """

    def __init__(self, frame_diff_threshold=1):
        super().__init__()
        self.frame_diff_threshold = frame_diff_threshold
        self.prev_frame = None

    def detect(self, frame, roi_bound=None, preprocessing=None):
        """
        Identifies the tracked object in the image ``frame``
        by comparing the difference with the previous frames. All the
        pixels differing by more than frame_diff_threshold will be
        considered part of the moving object.

        Parameters
        ----------
        frame : np.ndarray
            Image containing the object to be tracked
        roi_bound : tuple, optional
            Coordinates of the region of interest of the frame. The
            expected format if a tuple with the form (xmin, xmax, ymin,
            ymax). If passed the algorithm will crop this region of the
            frame and will proceed only in this region, providing the
            estimations refered to this region instead of the whole
            image, by default None.
        preprocessing : func
            A function to be applied to the frame (Or cropped version
            of it if roi_bound is passed) before detecting the object
            on it, by default None.

        Returns
        -------
        np.ndarray
            A binary version of ``frame`` where elements with value
            ``0`` indicate the absence of object and ``1`` the precense
            of the object.
        tuple
            x, y coordinates of the centroid of the object in the image.
        """

        if self.prev_frame is None:
            self.prev_frame = frame.copy()

        # Make a preprocessed (and copied) version of the frame
        cframe = self.preprocess(frame, roi_bound, preprocessing)
        prev_frame = self.preprocess(self.prev_frame, roi_bound, preprocessing)

        diff = cv2.absdiff(cframe, prev_frame)

        # Convert image to grayscale image
        gray_image = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale image to binary image
        mask = cv2.inRange(gray_image, self.frame_diff_threshold, 255)

        # Compute the centroid of the pixels over threshold
        centroid = self.get_centroid(mask)

        # Update previous image cache
        self.prev_frame = frame.copy()

        # Convert the grayscale image to binary image
        return mask, centroid


class BackgroundSubtraction(TrackingAlgorithm):
    """
    Identifies the position of an object by subtracting a known
    background.

    Parameters
    ----------
    background : np.ndarray
        Image containing the actual background of the scene where the
        images were taken. This algorithm will detect as an object of
        interest everything that differs from the background.
    background_threshold : int, optional
        Minimum difference (in terms of pixel intensity) among current
        image and background to consider that pixel as part of a moving
        object, by default 1.
    """

    def __init__(self, background, background_threshold):
        super().__init__()
        self.background_threshold = background_threshold
        self.background = background

    def detect(self, frame, roi_bound=None, preprocessing=None):
        """
        Identifies the tracked object in the image ``frame``
        by comparing the difference with the background. All the pixels
        differing by more than background_threshold will be considered
        part of the moving object.

        Parameters
        ----------
        frame : np.ndarray
            Image containing the object to be tracked
        roi_bound : tuple, optional
            Coordinates of the region of interest of the frame. The
            expected format if a tuple with the form (xmin, xmax, ymin,
            ymax). If passed the algorithm will crop this region of the
            frame and will proceed only in this region, providing the
            estimations refered to this region instead of the whole
            image, by default None.
        preprocessing : func
            A function to be applied to the frame (Or cropped version
            of it if roi_bound is passed) before detecting the object
            on it, by default None.

        Returns
        -------
        np.ndarray
            A binary version of ``frame`` where elements with value
            ``0`` indicate the absence of object and ``1`` the precense
            of the object.
        tuple
            x, y coordinates of the centroid of the object in the image.
        """

        # Make a preprocessed (and copied) version of the frame
        cframe = self.preprocess(frame, roi_bound, preprocessing)
        backgrn_roi = self.preprocess(self.background, roi_bound, preprocessing)

        diff = cv2.absdiff(cframe, backgrn_roi)

        # Convert image to grayscale image
        gray_image = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale image to binary image
        mask = cv2.inRange(gray_image, self.background_threshold, 255)

        # Compute the centroid of the pixels over threshold
        centroid = self.get_centroid(mask)

        # Convert the grayscale image to binary image
        return mask, centroid


class TemplateMatching(TrackingAlgorithm):
    """
    Identifies the position of an object by correlating with a template.

    Parameters
    ----------
    template : np.ndarray
        Image containing a template of a tipical image of the object
        being tracked. This algorithm will detect as an object of
        interest the point with higher correlation between the template
        and the image.
    threshold : float, optional
        Minimum value of correlation to be considered as a match, by
        default 0.8.
    """

    def __init__(self, template, threshold):
        super().__init__()
        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.threshold = threshold

        self.w, self.h = self.template.shape[::-1]

    def detect(self, frame, roi_bound=None, preprocessing=None):
        """
        Identifies the tracked object in the image ``frame``
        by comparing each region with a template. The region with higher
        correlation will be selected as the current position of the
        object.

        Parameters
        ----------
        frame : np.ndarray
            Image containing the object to be tracked
        roi_bound : tuple, optional
            Coordinates of the region of interest of the frame. The
            expected format if a tuple with the form (xmin, xmax, ymin,
            ymax). If passed the algorithm will crop this region of the
            frame and will proceed only in this region, providing the
            estimations refered to this region instead of the whole
            image, by default None.
        preprocessing : func
            A function to be applied to the frame (Or cropped version
            of it if roi_bound is passed) before detecting the object
            on it, by default None.

        Returns
        -------
        np.ndarray
            A binary version of ``frame`` where elements with value
            ``0`` indicate the absence of object and ``1`` the precense
            of the object.
        tuple
            x, y coordinates of the centroid of the object in the image.
        """

        # Make a preprocessed (and copied) version of the frame
        cframe = self.preprocess(frame, roi_bound, preprocessing)

        # Convert image to grayscale image
        gray_img = cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)

        # Compute the correlation
        res = cv2.matchTemplate(gray_img, self.template, cv2.TM_CCOEFF_NORMED)

        # Get the point of max correlation
        pt = np.unravel_index(res.argmax(), res.shape)

        # Compute the centroid of the region with max correlation
        centroid = None
        if res[pt] > self.threshold:
            centroid = (int(pt[1] + self.w/2), int(pt[0] + self.h/2))


        # Convert the grayscale image to binary image
        mask = None

        # Convert the grayscale image to binary image
        return mask, centroid


class OpticalFlow(TrackingAlgorithm):
    """
    This class implements optical flow based on
    Gunner Farneback's algorithm. A section of the
    frame is selected and tracked using dense optical flow.

    Parameters
    ----------
    threshold : float
        Minimum value for the magnitude of optical flow to be considered
        part of the motion.
    buffer_size : int, optional
        Indicates how many frames in the past the algorithm is going to
        look before computing the optical flow, by default 1.
    """

    def __init__(self, threshold, buffer_size=1):
        super().__init__()
        self.threshold = threshold
        self.previous_frames = []

        assert buffer_size > 0

        self.buffer_size = buffer_size


    def detect(self, frame, roi_bound=None, preprocessing=None):
        """
        Identifies the tracked object in the image ``frame``
        by tracking the motion of a region using optical flow.

        Parameters
        ----------
        frame : np.ndarray
            Image containing the object to be tracked
        roi_bound : tuple, optional
            Coordinates of the region of interest of the frame. The
            expected format if a tuple with the form (xmin, xmax, ymin,
            ymax). If passed the algorithm will crop this region of the
            frame and will proceed only in this region, providing the
            estimations refered to this region instead of the whole
            image, by default None.
        preprocessing : func
            A function to be applied to the frame (Or cropped version
            of it if roi_bound is passed) before detecting the object
            on it, by default None.

        Returns
        -------
        np.ndarray
            A binary version of ``frame`` where elements with value
            ``0`` indicate the absence of object and ``1`` the precense
            of the object.
        tuple
            x, y coordinates of the centroid of the object in the image.
        """

        if len(self.previous_frames) == self.buffer_size:

            cframe = self.preprocess(
                frame=frame,
                roi_bound=roi_bound,
                preprocessing=preprocessing
            )
            pframe = self.preprocess(
                frame=self.previous_frames[-1],
                roi_bound=roi_bound,
                preprocessing=preprocessing
            )

            cframe = cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)
            pframe = cv2.cvtColor(pframe, cv2.COLOR_BGR2GRAY)

            diff = cv2.calcOpticalFlowFarneback(
                prev=pframe,
                next=cframe,
                flow=None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            mag, _ = cv2.cartToPolar(diff[..., 0], diff[..., 1])

            # Convert the grayscale image to binary image
            mask = cv2.inRange(mag, self.threshold, 255)

            # Compute the centroid of the pixels over threshold
            centroid = self.get_centroid(mask)

            self.previous_frames.pop(0)
        else:
            mask, centroid = None, None

        # Store the current frame
        self.previous_frames.append(frame.copy())

        return mask, centroid
