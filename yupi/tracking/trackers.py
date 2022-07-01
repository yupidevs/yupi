import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np

from yupi.tracking.algorithms import TrackingAlgorithm, _resize_frame
from yupi.tracking.undistorters import Undistorter
from yupi.trajectory import Trajectory
from yupi.transformations import add_moving_FoR
from yupi.transformations._affine_estimator import AffineParams, _get_affine

# pylint: disable=protected-access


Centroid = Tuple[int, int]
"""Centroid of a tracked object: x, y."""

Bounds = Tuple[int, int, int, int]
"""Bounds of a frame: x_min, x_max, y_min, y_max."""


class ROI:
    """
    Region of interest.

    Region that can be tracked by the algorithms throughout the sequence
    of image frames.

    Parameters
    ----------
    size : Tuple[float, float]
        Size of the region of interest.

        If both tuple's values are grater than 1 then they are rounded
        and taken as pixels. Otherwise, if both values are less than 1,
        the size is taken relative to the video frame size.
    init_mode : str, optional
        ROI's initialization mode, by default 'manual'.

        Defines the way ROI initial position is setted.

        The ``init_mode`` parameter can be manual or center. These
        modes are stored in ``ROI.MANUAL_INIT_MODE`` and
        ``ROI.CENTER_INIT_MODE``.
    scale : float, optional
        Scale of the sample frame to set ROI initial position if
        ``init_method`` is set to ``'manual'``, by default 1.

    Attributes
    ----------
    width : float
        Width of the ROI.

        If the width value is between 0 and 1 then this is taken
        relative to the frames. Otherwise it is a rounded value and
        taken as pixels.
    height : float
        Height of the ROI.

        If the height value is between 0 and 1 then this is taken
        relative to the frames. Otherwise it is a rounded value and
        taken as pixels.
    init_mode : str
        ROI's initialization mode.
    scale : float
        Scale of the sample frame to set ROI initial position if
        ``init_method`` is set to ``'manual'``.

    Examples
    --------
    >>> ROI((120, 120), ROI.MANUAL_INIT_MODE)
    ROI: size=(120, 120) init_mode=manual scale=1

    Raises
    ------
    ValueError
        If any size value is negative.
    ValueError
        If one of the size value is grater than 1 and the other is less
        than 1.
    ValueError
        If ROI initialization mode is neither ``'manual'`` or
        ``'center'``.
    """

    MANUAL_INIT_MODE = "manual"
    """Manual initialization mode for the ROI"""
    CENTER_INIT_MODE = "center"
    """Center initialization mode for the ROI"""

    def __init__(
        self,
        size: Tuple[float, float],
        init_mode: str = MANUAL_INIT_MODE,
        scale: float = 1,
    ):

        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("ROI's size values must be positives")

        # TODO: check for a more pythonic way to do this comprobation
        if (size[0] < 1 and size[1] > 1) or (size[0] > 1 and size[1] < 1):
            raise ValueError(
                "Size values must be between 0 and 1 both or "
                "integers greater than 0 both"
            )

        if init_mode not in (ROI.CENTER_INIT_MODE, ROI.MANUAL_INIT_MODE):
            raise ValueError(f"ROI '{init_mode}' initialization mode unknown")

        if scale < 0:
            raise ValueError("ROI scale must be non negative")

        self.width, self.height = size
        self.init_mode = init_mode
        self.scale = scale

        self._prev_centroid: Centroid
        self._centroid: Centroid
        self._global_height: int
        self._global_width: int

    def __repr__(self):
        return (
            "ROI: size=({self.width}, {self.height}) "
            "init_mode={self.init_mode} scale={self.scale}"
        )

    def _recenter(self, centroid: Optional[Centroid]) -> None:
        """
        Recenters ROI position.

        Parameters
        ----------
        centroid : tuple of int
            New center of the ROI.
        """

        # Get the centroid refered to the roi
        if centroid is not None:
            roi_cx, roi_cy = centroid

            # Get the centroid refered to the full image
            c_x = self._prev_centroid[0] - int(self.width / 2) + roi_cx
            c_y = self._prev_centroid[1] - int(self.height / 2) + roi_cy

            c_x = min(c_x, self._global_width)
            c_x = max(c_x, 0)
            c_y = min(c_y, self._global_height)
            c_y = max(c_y, 0)

            self._centroid = (c_x, c_y)

        else:
            self._centroid = self._prev_centroid

    def _get_bounds(self, prev: bool = False) -> Bounds:
        """
        ROI's bounds.

        Calculates the ROI's bounds according to its center, width,
        height and the global bounds.

        Parameters
        ----------
        prev : bool
            Use previous roi center instead of current

        Returns
        -------
        xmin : int
            Mnimum bound on X axis.
        xmax : int
            Maximun bound on X axis.
        ymin : int
            Mnimum bound on Y axis.
        ymax : int
            Maximum bound on Y axis.
        """

        if prev:
            c_x, c_y = self._prev_centroid
        else:
            c_x, c_y = self._centroid

        half_width, half_height = int(self.width / 2), int(self.height / 2)
        xmin = max(c_x - half_width, 0)
        xmax = min(c_x + half_width, self._global_width)
        ymin = max(c_y - half_height, 0)
        ymax = min(c_y + half_height, self._global_height)
        return xmin, xmax, ymin, ymax

    def _center_init(self, frame: np.ndarray) -> Centroid:
        """
        Initialize ROI using center initialization mode.

        Parameters:
        frame : np.ndarray
            Frame used as reference to initialize ROI position at its
            center.

        Returns
        -------
        tuple of int
            Center of the ROI.
        """

        self._global_height, self._global_width = frame.shape[:2]
        self._centroid = self._global_width // 2, self._global_height // 2
        return self._centroid

    # TODO: check for 'win2_name' utility. Maybe it should be 'ROI' as
    # Default so there is no need to pass it as a parameter
    def _manual_init(self, frame: np.ndarray, name: str) -> Centroid:
        """
        Initialize ROI using manual initialization mode.

        Parameter
        ---------
        frame : np.ndarray
            Frame used as reference to initialize ROI position manually.
        name : str
            Name of the tracking object.

        Returns
        -------
        tuple of int
            Center of the ROI.
        """

        win1_name = (
            "Initialization of trackers: Click on the initial "
            f"position of: {name.upper()}"
        )
        logging.info("Open the video window to select %s's center", name)

        self._global_height, self._global_width = frame.shape[:2]

        frame_ = _resize_frame(frame, scale=self.scale)
        cv2.imshow(win1_name, frame_)

        roi_initialized = False
        # Callback handler to manually set the roi
        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Global roi center coordinates
                self._centroid = int(x / self.scale), int(y / self.scale)

                # Copy of true frame and its resized version
                img_ = frame_.copy()

                # Draw a circle in the selected pixel
                cv2.circle(img_, (x, y), 3, (0, 255, 255), 1)
                xmin, xmax, ymin, ymax = self._get_bounds()
                pt1 = (int(xmin * self.scale), int(ymin * self.scale))
                pt2 = (int(xmax * self.scale), int(ymax * self.scale))
                cv2.rectangle(img_, pt1, pt2, (0, 255, 255), 1)

                msg = "ROI initialized. Press any key to continue."

                cv2.putText(
                    img_, msg, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                )

                cv2.imshow(win1_name, img_)

                nonlocal roi_initialized
                roi_initialized = True
                logging.info("ROI initialized. Press any key to continue.")

        cv2.setMouseCallback(win1_name, on_click)
        while not roi_initialized:
            cv2.waitKey(0)
            logging.info(
                "Waiting for ROI initialization. Please "
                "click on the center of the %s's ROI on "
                "the video window.",
                name,
            )

        return self._centroid

    def _check_roi_init(self, name: str) -> bool:
        """
        Checks for ROI initialization.

        Parameter
        ---------
        name : str
            Name of the tracking object.

        Returns
        -------
        bool
            Whether or not the ROI is initialized.
        """

        if not self._prev_centroid[0]:
            logging.error("ROI was not initialized in %s", name)
            return False

        cv2.destroyAllWindows()
        logging.info("ROI initialized in %s", name)
        return True

    def _initialize(self, name: str, first_frame: np.ndarray) -> bool:
        """
        Initialize ROI.

        Parameters
        ----------
        name : str
            Name of the tracking object.
        first_frame : np.ndarray
            First frame of the video.

            If ROI's initialization mode is set to ``'manual'`` this
            frame will be shown to select the tracking object center.

        Returns
        -------
        bool
            Whether or not the ROI was initialized.
        """

        height, weight = first_frame.shape[:2]
        if self.width <= 1:
            self.width *= weight
        if self.height <= 1:
            self.height *= height

        # Initialize ROI coordinates manually by user input
        if self.init_mode == ROI.MANUAL_INIT_MODE:
            self._centroid = self._manual_init(first_frame, name)
        else:
            self._centroid = self._center_init(first_frame)

        self._prev_centroid = self._centroid
        return self._check_roi_init(name)

    def _crop(self, frame: np.ndarray, prev: bool = False) -> np.ndarray:
        """
        Crops a frame according to the ROI's bounds.

        Parameters
        ----------
        frame : np.ndarray
            Frame that will be cropped.
        prev : bool
            Use previous roi center instead of current

        Returns
        -------
        window : np.ndarray
            Cropped part of the frame.
        """

        self._global_height, self._global_width = frame.shape[:2]
        # Bounds of the roi
        xmin, xmax, ymin, ymax = self._get_bounds(prev)
        window = frame[ymin:ymax, xmin:xmax, :]
        return window.copy()


class ObjectTracker:
    """
    Tracks an object inside a ROI according to a tracking algorithm.

    Parameters
    ----------
    name : str
        Name of the tracked object.
    algorithm : TrackingAlgorithm
        Algorithm used to track the object.
    roi : ROI
        Region of interest where the object will be tracked.
    preprocessing : Callable[[np.ndarray], np.ndarray], optional
        Preprocessing function aplied to the frame before being used by
        the algorithm.

    Attributes
    ----------
    name : str
        Name of the tracked object.
    algorithm : TrackingAlgorithm
        Algorithm used to track the object.
    roi : ROI
        Region of interest where the object will be tracked.
    history : List[Centroid]
        ROI's position in every frame of the video.
    preprocessing : Optional[Callable[[np.ndarray], np.ndarray]]
        Preprocessing function aplied to the frame before being used by
        the algorithm.
    """

    def __init__(
        self,
        name: str,
        algorithm: TrackingAlgorithm,
        roi: ROI,
        preprocessing: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.name = name
        self.roi = roi
        self.algorithm = algorithm
        self.preprocessing = preprocessing
        self.history: List[Centroid] = []
        self.mask: np.ndarray

    def _init_roi(self, frame: np.ndarray) -> bool:
        return self.roi._initialize(self.name, frame)

    def _track(self, frame: np.ndarray):
        """
        Tracks the center of the object.

        Given a new frame, the center of the object inside the ROI is
        recalculated using the selected algorithm.

        Parameters
        ----------
        frame : np.ndarray
            Frame used by the algorithm to detect the tracked object's
            new center.
        """

        # Get only the ROI from the current frame
        roi_bound = self.roi._get_bounds()

        # Detect the object using the tracking algorithm
        self.mask, centroid = self.algorithm.detect(
            frame, roi_bound, self.preprocessing
        )

        # Update the roi center using current ant coordinates
        self.roi._recenter(centroid)

        # Update data
        self.history.append(self.roi._centroid)


class CameraTracker:
    """
    Tracks the camera movement.

    Parameters
    ----------
    roi : ROI
        Region of interest where the background changes will be
        detected.

    Attributes
    ----------
    roi : ROI
        Region of interest where the background changes will be
        detected.
    affine_params_history : List[AffineParams]
        History of all the affine parameters
    """

    def __init__(self, roi: ROI):
        self.affine_params_history: List[AffineParams] = []
        self.mse: List[float] = []
        self.roi = roi
        self.features: Any

    def _init_roi(self, prev_frame: np.ndarray) -> bool:
        return self.roi._initialize("Camera", prev_frame)

    # Track the floor
    def _track(
        self, prev_frame: np.ndarray, frame: np.ndarray, ignored_regions: List[Bounds]
    ) -> bool:
        """
        Tracks the camera movements according to the changing background
        inside the ROI.

        Parameters
        ----------
        prev_frame, frame : np.ndarray
            Frames used to detect background movement.
        igonerd_regions : List[Bounds]
            Tracked object's boundaries.

            Tracked object's does not form part of the background so
            they should be ignored.

        Returns
        -------
        bool
            Whether or not good points were found or sucessfully
            tracked.
        """

        # Initialize a mask of what to track
        height, weight = frame.shape[:2]
        mask = 255 * np.ones((height, weight), dtype=np.uint8)

        # Mask pixeles inside every ROIs
        for x_0, x_f, y_0, y_f in ignored_regions:
            mask[y_0:y_f, x_0:x_f] = 0

        p_good, affine_params, err = _get_affine(
            img1=prev_frame, img2=frame, region=self.roi._get_bounds(), mask=mask
        )
        self.features = p_good[1:]

        if err is None:
            return False

        self.affine_params_history.append(affine_params)
        self.mse.append(err)

        return True


class TrackingScenario:
    """
    Controls all the tracking process along the video.

    Parameters
    ----------
    object_trackers : list
        Trackers of all the objects.
    camera_tracker : CameraTracker
        Tracker used to detect camera movements, by default None.
    undistorter : Undistorter
        Undistorted used to correct each video frame, by default None.
    preview_scale : float
        Scale of the video preview, by default 1.0.
    auto_mode : bool
        If True the video is processed auomtically otherwise it's
        processed manually, by default True.

        If the video is processed manually, pressing ``ENTER`` key is
        necessary in every frame to continue.

        This mode can be changed in the middle of the processing by
        pressing ``M`` key.

    Attributes
    ----------
    object_trackers : list
        Trackers of all the objects
    camera_tracker : CameraTracker
        Tracker used to detect camera movements.
    undistorter : Undistorter
        Undistorted used to correct each video frame.
    preview_scale : float
        Scale of the video preview.
    auto_mode : bool
        If True the video is processed auomtically otherwise it's
        processed manually, by default True.

        If the video is processed manually, pressing ``Enter`` key is
        necessary in every frame to continue.

        This mode can be changed in the middle of the processing by
        pressing ``M`` key.
    """

    def __init__(
        self,
        object_trackers: List[ObjectTracker],
        camera_tracker: Optional[CameraTracker] = None,
        undistorter: Optional[Undistorter] = None,
        preview_scale: float = 1,
        auto_mode: bool = True,
    ):
        self.object_trackers = object_trackers
        self.camera_tracker = camera_tracker
        self.undistorter = undistorter
        self.preview_scale = preview_scale
        self.auto_mode = auto_mode
        self._enabled = True
        self._iteration_counter = 0

        self.video_path: str
        self.cap: Any
        self.frame_count: int
        self.fps: int
        self.width: int
        self.height: int
        self.dim: Tuple[int, int]
        self.prev_frame: np.ndarray
        self.first_frame: int
        self.last_frame: Optional[int] = None

    def _digest_video_path(self, video_path):
        if not Path.exists(Path(video_path)):
            raise ValueError(f"Path '{video_path}' does not exists")
        self.video_path = video_path

        # Create capture object
        self.cap = cv2.VideoCapture(video_path)

        # Total number of frames in the video file
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Frames per seconds
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Frame width
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Frame height
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.dim = (self.width, self.height)

        self.first_frame = 0

    def _undistort(self, frame):
        if self.undistorter:
            frame = self.undistorter.fix(frame)
        return frame

    def _show_frame(self, frame, show_frame_id=True):
        # CXY, region, features, frame_numb, mask
        frame = frame.copy()

        # Draw region in which features are detected
        if self.camera_tracker:
            x_0, x_f, y_0, y_f = self.camera_tracker.roi._get_bounds()

            cv2.putText(
                img=frame,
                text="Camera Tracking region",
                org=(x_0 + 5, y_f - 5),
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                fontScale=1.2,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            cv2.rectangle(frame, (x_0, y_0), (x_f, y_f), (0, 0, 255), 2)
            p_2, p_3 = self.camera_tracker.features
            # Draw detected and estimated features
            for p2_, p3_ in zip(p_2, p_3):
                x_2, y_2 = np.rint(p2_).astype(np.int32)
                x_3, y_3 = np.rint(p3_).astype(np.int32)

                cv2.circle(frame, (x_2, y_2), 3, (0, 0, 0), -1)
                cv2.circle(frame, (x_3, y_3), 3, (0, 255, 0), -1)

        for obj_tracker in self.object_trackers:
            # TODO: Do this better:
            # Alter the blue channel in ant-related pixels
            window = obj_tracker.roi._crop(frame, prev=True)
            if obj_tracker.mask is not None:
                window[:, :, 0] = obj_tracker.mask

            # Draw a point over the roi center and draw bounds
            x_1, x_2, y_1, y_2 = obj_tracker.roi._get_bounds()
            cv2.circle(frame, obj_tracker.roi._centroid, 5, (255, 255, 255), -1)
            cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), (0, 255, 255), 2)
            cv2.putText(
                img=frame,
                text=obj_tracker.name,
                org=(x_1 + 5, y_2 - 5),
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                fontScale=1.2,
                color=(0, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        if show_frame_id:
            height, weight = frame.shape[:2]
            frame_id = self._iteration_counter + self.first_frame
            x, y = int(0.02 * weight), int(0.05 * height)
            cv2.putText(
                img=frame,
                text=str(frame_id),
                org=(x, y),
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                fontScale=1.2,
                color=(0, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        frame = _resize_frame(frame, self.preview_scale)
        cv2.imshow("yupi processing window", frame)

    def _create_ui(
        self,
        img: np.ndarray,
        t_name: str,
        current_tracker: int,
        total_trackers: int,
        roi: ROI,
    ):
        imgc = img.copy()
        imgc = _resize_frame(imgc, roi.scale)
        height = img.shape[0] * roi.scale
        weight = img.shape[1] * roi.scale
        h_pad = 0.2
        w_pad = 0.15

        imgc = cv2.blur(imgc, (5, 5))

        box = imgc[
            int(height * h_pad) : int(height - height * h_pad),
            int(weight * w_pad) : int(weight - weight * w_pad),
            :,
        ]
        threshold = 180

        box[:, :] += np.array([threshold, threshold, threshold], dtype="uint8")
        box[box < threshold] = 255

        bshape = box.shape
        boxw = bshape[1]
        boxh = bshape[0]

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.0008333 * weight + 0.0333333
        color = (50, 50, 50)
        thickness = int(font_scale + 1)

        text_lines = [
            "Your tracking scenario is almost ready",
            "Let's initialize your trackers",
            "Next, you will have to click on the initial",
            f"position of the tracker {t_name.upper()}",
            "Press any key to continue...",
            f"Trackers Initialized: {current_tracker}/{total_trackers}",
        ]

        l = int(0.0396825 * boxw + 2.222222)

        def put_text(img: np.ndarray, text: str, pos: Tuple[float, float]):
            return cv2.putText(
                img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA
            )

        box = put_text(box, text_lines[0], (l, l + l))
        box = put_text(box, text_lines[1], (l, l + 2 * l))
        box = put_text(box, text_lines[2], (l, l + 4 * l))
        box = put_text(box, text_lines[3], (l, l + 5 * l))
        box = put_text(box, text_lines[4], (l, l + 7 * l))
        box = put_text(box, text_lines[5], (l, boxh - l))
        return imgc

    def _first_iteration(self, start_frame):
        # Start processing frams at the given index
        if start_frame:
            self.first_frame = start_frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Capture the first frame to process
        _, prev_frame = self.cap.read()

        # Correct spherical distortion
        self.prev_frame = self._undistort(prev_frame)

        # Initialize the roi of all the trackers
        for i, obj_tracker in enumerate(self.object_trackers):
            if obj_tracker.roi.init_mode == "manual":
                tracker_name = obj_tracker.name
                ui = self._create_ui(
                    self.prev_frame,
                    tracker_name,
                    i,
                    len(self.object_trackers),
                    obj_tracker.roi,
                )
                cv2.imshow(
                    "Initialization of trackers: Press any key to start with "
                    f"tracker: {tracker_name.upper()}",
                    ui,
                )
                cv2.waitKey(-1)
                cv2.destroyAllWindows()
            retval = obj_tracker._init_roi(self.prev_frame)
            if not retval:
                return retval

        # Initialize the region of the camera tracker
        if self.camera_tracker:
            self.camera_tracker._init_roi(self.prev_frame)

        # Increase the iteration counter
        self._iteration_counter += 1

        logging.info("All trackers were initialized")
        return True

    def _keyboard_controller(self):
        # Keyboard events
        wait_key = 0 if not self.auto_mode else 10

        key = cv2.waitKey(wait_key) & 0xFF
        if key == ord("m"):
            self.auto_mode = not self.auto_mode

        elif key == ord("q"):
            self._enabled = False

        elif key == ord("e"):
            exit()

    def _regular_iteration(self):
        # Get current frame and ends the processing when no more frames are
        # detected

        frame_id = self._iteration_counter + self.first_frame
        if self.last_frame is not None and frame_id >= self.last_frame:
            return False, True

        ret, frame = self.cap.read()
        if not ret:
            logging.info("All frames were processed")
            return False, True

        # Correct spherical distortion
        frame = self._undistort(frame)

        # ROI Arrays of tracking objects
        roi_array = []

        # Track every object and save past and current ROIs
        for otrack in self.object_trackers:
            roi_array.append(otrack.roi._get_bounds())
            otrack._track(frame)
            roi_array.append(otrack.roi._get_bounds())

        if self.camera_tracker:
            ret = self.camera_tracker._track(self.prev_frame, frame, roi_array)

        if not ret:
            msg = f"CameraTracker - No matrix was estimated (Frame {frame_id})"
            logging.error(msg)
            return False, False

        # Display the full image with the ant in blue (TODO: Refactor this to
        # Make more general)
        self._show_frame(frame)

        for otrack in self.object_trackers:
            otrack.roi._prev_centroid = otrack.roi._centroid

        # Save current frame and ROI center as previous for next iteration
        self.prev_frame = frame.copy()

        # Call the keyboard controller to handle key interruptions
        self._keyboard_controller()

        # Increase the iteration counter
        self._iteration_counter += 1

        return True, False

    def _release_cap(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def _tracker2trajectory(self, tracker, pix_per_m):
        dt = 1 / self.fps
        traj_id = tracker.name
        x, y = map(list, zip(*tracker.history))
        x_arr = np.array(x) / pix_per_m
        y_arr = -1 * np.array(y) / pix_per_m
        return Trajectory(x=x_arr, y=y_arr, dt=dt, traj_id=traj_id)

    def _export_trajectories(self, pix_per_m):
        t_list = []
        reference = None
        # Extract camera reference
        if self.camera_tracker:
            affine_params = np.array(self.camera_tracker.affine_params_history)
            theta, t_x, t_y, _ = affine_params.T
            t_x, t_y = t_x / pix_per_m, t_y / pix_per_m
            # Invert axis
            theta *= -1
            t_y *= -1
            reference = theta, t_x, t_y
        # Output the trajectory of each tracker
        for otrack in self.object_trackers:
            t = self._tracker2trajectory(otrack, pix_per_m)
            if self.camera_tracker:
                assert reference is not None
                t = add_moving_FoR(t, reference, new_traj_id=t.traj_id)
            t_list.append(t)
        return t_list

    def track(
        self,
        video_path: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        pix_per_m: int = 1,
    ) -> Tuple[bool, Optional[List[Trajectory]]]:
        """
        Starts the tracking process.

        Parameters
        ----------
        video_path : str
            Path of the video used to track the objects.
        start_frame : int, optional
            Initial frame in which starts the processing, by default 0.
        end_frame : Optional[int]
            Last frame being processed, if nothing is passed all frames
            until the end of the video will be processed, by default
            None.
        pix_per_m : int, optional
            Pixel per meters, by default 1.

            This value is used to readjuts the trajectories points to a
            real scale.

        Returns
        -------
        bool
            Whether or not the tracking process ended succefully.
        List[Trajectory]
            List of all the trajectories extracted in the tracking
            process.
        """

        if end_frame is not None and end_frame > start_frame:
            self.last_frame = int(end_frame)

        self._digest_video_path(video_path)

        end = False
        retval = False
        if self._iteration_counter == 0:
            retval = self._first_iteration(start_frame)
            if not retval:
                return retval, None

        logging.info("Processing frames")
        while self._enabled:
            retval, end = self._regular_iteration()
            if not retval:
                break

        if end:
            retval = True

        self._release_cap()
        trajectories = self._export_trajectories(pix_per_m)
        return retval, trajectories
