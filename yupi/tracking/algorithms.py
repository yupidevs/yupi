import abc
import cv2
import numpy as np


def resize_frame(frame, scale=1):
    h, w = frame.shape[:2]
    w_, h_ = int(scale * w), int(scale * h)
    short_frame = cv2.resize(frame, (w_, h_), interpolation=cv2.INTER_AREA)
    return short_frame


def change_colorspace(image, color_space):
    if color_space == 'BGR':
        return image.copy()
    if color_space == 'GRAY':
        return cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    if color_space == 'HSV':
        return cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)


class BackgroundEstimator():
    """
    This class provides static methods to determine the background in image
    sequences. It estimates the temporal median of the sequence.
    """
    def __init__(self):
        pass

    def from_video(video_path, samples, start_in=0):
        """
        This method takes a video indicated by ``video_path`` and uniformely 
        take a number of image samples according to the parameter ``samples``.
        Then, it computes the temporal median of the images in order to 
        determine de background.

        Parameters
        ----------
        video_path : str
            Path to the video file
        samples : int
            Number of samples to get from the video
        start_in : int, optional
            If passed, the method will start sampling after the frame indicated
            by this value, by default 0
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
            ret, frame = cap.read()
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
        else:
            print('[ERROR] Nothing was over threshold')
            return None

    @abc.abstractmethod
    def detect(self, current_chunk, previous_chunk=None):
        """
        Abstract method that is implemented on inheriting classes.
        It should compute the location (in the image ``current_chunck``)
        of the object being tracked.


        Parameters
        ----------
        current_chunk : np.ndarray
            Image where the algorithm must identify the object
        previous_chunk : np.ndarray, optional
            Previous image where the algorithm already identified the
            object, by default None.
        """


# TODO: Fix this to emulate the previous Trackingalgorithm including this:
# ant_ratio = ant_pixels / (roi_width * roi_heigh)
# approximate ratio of the ant compare to the roi
class IntensityMatching(TrackingAlgorithm):
    """
    Identifies the position of an object by thresholding the pixel
    intensities of grayscale images.

    Parameters
    ----------
    min_val : int, optional
        Minimum value of pixel intensity to be considered as part of
        the object, by default 0.
    max_val : int, optional
        Maximum value of pixel intensity to be considered as part of
        the object, by default 255.
    max_pixels : int, optional
        If this parameter is passed, the algoritm will stop searching for
        candidate pixels after reaching a count equal to this value,
        by default None.
        """
    def __init__(self, min_val=0, max_val=255, max_pixels=None):

        super(IntensityMatching, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.max_pixels = max_pixels

    def detect(self, current_chunk, previous_chunk=None):
        """
        Identifies the tracked object in the image ``current_cunk``
        by thresholding its grayscale version using the parameters
        defined when the object was constructed.

        Parameters
        ----------
        current_chunk : np.ndarray
            Image containing the object to be tracked

        Returns
        -------
        tuple
             * mask: np.ndarray (a binary version of ``current_chunk`` where
               elements with value ``0`` indicate the absence of object
               and ``1``
               the precense of the object.
             * centroid: tuple (x, y coordinates of the centroid of the object
               in the image)

        """
        # Convert image to grayscale image
        gray_image = cv2.cvtColor(current_chunk.copy(), cv2.COLOR_BGR2GRAY)

        if self.max_pixels:
            # Obtain image histogram
            ys = cv2.calcHist([gray_image], [0], None, [256], [0, 256],
                              accumulate=True)

            # Compute image total pixel count
            x, y = gray_image.shape
            total = x * y

            # Compute an adaptative threshold according the ratio of
            # Darkest pixels
            max_threshold = self.max_val
            for i in range(self.min_val, self.max_val):
                if ys[i] > self.max_pixels:
                    max_threshold = i
                    break
            self.max_val = max_threshold

        mask = cv2.inRange(gray_image, self.min_val, self.max_val)
        centroid = self.get_centroid(mask)
        # Convert the grayscale image to binary image
        return mask, centroid


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
        If this parameter is passed, the algoritm will stop searching for
        candidate pixels after reaching a count equal to this value,
        by default None.
    """

    def __init__(self, lower_bound=(0, 0, 0), upper_bound=(255, 255, 255),
                 color_space='BGR', max_pixels=None):
        super(ColorMatching, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.color_space = color_space
        self.max_pixels = max_pixels

    def detect(self, current_chunk, previous_chunk=None):
        """
        Identifies the tracked object in the image ``current_cunk``
        by thresholding it using the bound parameters defined when
        the object was constructed.

        Parameters
        ----------
        current_chunk : np.ndarray
            Image containing the object to be tracked

        Returns
        -------
        tuple
             * mask: np.ndarray (a binary version of ``current_chunk`` where
               elements with value ``0`` indicate the absence of object and ``1``
               the precense of the object.
             * centroid: tuple (x, y coordinates of the centroid of the object
               in the image)

        """
        # Convert image to desired colorspace
        copied_image = change_colorspace(current_chunk, self.color_space)

        mask = cv2.inRange(copied_image, self.lower_bound, self.upper_bound)
        centroid = self.get_centroid(mask)
        # Convert the grayscale image to binary image
        return mask, centroid


class FrameDifferencing(TrackingAlgorithm):
    """
    Identifies the position of an object by comparison between consecutive 
    frames

    Parameters
    ----------
    frame_diff_threshold : int, optional
        If this parameter is passed, the algoritm will stop searching for
        candidate pixels after reaching a count equal to this value,
        by default 1.
    """

    def __init__(self, frame_diff_threshold=1):
        super(FrameDifferencing, self).__init__()
        self.frame_diff_threshold = frame_diff_threshold

    def detect(self, current_chunk, previous_chunk=None):
        """
        Identifies the tracked object in the image ``current_cunk``
        by comparing the difference with previous chunk. All the pixels
        differing by more than frame_diff_threshold will be considered
        part of the moving object.

        Parameters
        ----------
        current_chunk : np.ndarray
            Image containing the object to be tracked
        previous_chunk : np.ndarray
            Image containing the same region of current_chunk in a previous time
            

        Returns
        -------
        tuple
             * mask: np.ndarray (a binary version of ``current_chunk`` where
               elements with value ``0`` indicate the absence of object and 
               ``1`` the precense of the object.
             * centroid: tuple (x, y coordinates of the centroid of the object
               in the image)

        """
        print(current_chunk.shape, previous_chunk.shape)
        diff = cv2.absdiff(current_chunk, previous_chunk)

        # Convert image to grayscale image
        gray_image = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale image to binary image
        mask = cv2.inRange(gray_image, self.frame_diff_threshold, 255)

        # Compute the centroid of the pixels over threshold
        centroid = self.get_centroid(mask)

        # Convert the grayscale image to binary image
        return mask, centroid