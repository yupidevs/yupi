"""
This contains all the affine estimator related functions.
"""

import logging
from typing import Optional, Tuple

import cv2
import nudged
import numpy as np

AffineParams = Tuple[float, float, float, float]
"""Affine params: theta, t_x, t_y, scale."""

# ShiTomasi corner detection
FEATURE_PARAMS = dict(maxCorners=30, qualityLevel=0.6, minDistance=30, blockSize=100)

# Lucas Kanade optical flow
LK_PARAMS = dict(
    winSize=(20, 20),
    maxLevel=15,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 0.05),
)


def _rot_matrix(theta: float, inverse: bool = False) -> np.ndarray:
    rot_mat = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rot_mat.T if inverse else rot_mat


def _affine_matrix(
    theta: float, t_x: float, t_y: float, scale: float = 1.0, inverse: bool = False
) -> np.ndarray:
    rot_mat = _rot_matrix(theta, inverse)
    rot_mat = (scale * np.identity(2)) @ rot_mat
    shift = np.array([t_x, t_y])
    aff_mat = np.hstack([rot_mat, shift[None, :].T])
    return aff_mat


def _estimate_params(
    p_1: np.ndarray, p_2: np.ndarray
) -> Tuple[float, float, float, float]:
    transf = nudged.estimate(p_1, p_2)
    t_x, t_y = transf.get_translation()
    theta, scale = transf.get_rotation(), transf.get_scale()
    return theta, t_x, t_y, scale


def _get_p3(
    p_1: np.ndarray, theta: float, t_x: float, t_y: float, scale: float = 1.0
) -> np.ndarray:
    aff_mat = _affine_matrix(theta, t_x, t_y, scale)
    p_3 = np.array([aff_mat @ [x1, y1, 1] for (x1, y1) in p_1])
    return p_3


def _get_r(p_1: np.ndarray, p_2: np.ndarray) -> np.ndarray:
    x, y = (p_1 - p_2).T
    return np.sqrt(x**2 + y**2)


def _get_mask_r(r: np.ndarray, quantile: float = 1.5) -> np.ndarray:
    # Median and standard deviation of distance population
    r_median, r_std = np.median(r), np.std(r)
    # Cutoff used to filter far features points
    bound = max(1, quantile * r_std)
    # Get mask for nearby points
    mask = r < r_median + bound
    return mask


def _delete_far_points(
    p_1: np.ndarray,
    p_2: np.ndarray,
    p_3: Optional[np.ndarray] = None,
    quantile: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    # Distance amoung p1 and p2, or p2 and p3
    r = _get_r(p_1, p_2) if p_3 is None else _get_r(p_2, p_3)
    # Mask that filters outliers for a given quantile value
    mask = _get_mask_r(r, quantile)
    # Delete far points
    p_1, p_2 = p_1[mask], p_2[mask]
    return p_1, p_2


def _estimate_matrix(
    p_1: np.ndarray,
    p_2: np.ndarray,
    quantile1: Optional[float] = None,
    quantile2: Optional[float] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], AffineParams]:
    # Validate tracked features deletting outliers
    if quantile1 is not None:
        p_1, p_2 = _delete_far_points(p_1, p_2, quantile=quantile1)

    # Estimate matrix parameters and transformed features
    affine_params = _estimate_params(p_1, p_2)
    p_3 = _get_p3(p_1, *affine_params)

    # Delete outliers considering transformed features
    if quantile2 is not None:
        p_1, p_2 = _delete_far_points(p_1, p_2, p_3, quantile=quantile2)
        affine_params = _estimate_params(p_1, p_2)
        p_3 = _get_p3(p_1, *affine_params)

    points = p_1, p_2, p_3
    return points, affine_params


def _get_rmse(p_2: np.ndarray, p_3: np.ndarray) -> float:
    x, y = (p_2 - p_3).T
    return np.sqrt(np.mean(x**2 + y**2))


def _get_affine(
    img1: np.ndarray,
    img2: np.ndarray,
    region: Tuple[int, int, int, int],
    mask: Optional[np.ndarray] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], AffineParams, Optional[float]]:
    x_0, x_f, y_0, y_f = region

    # Get main regions
    img1_region = img1[y_0:y_f, x_0:x_f, :]
    img2_region = img2[y_0:y_f, x_0:x_f, :]

    if mask is not None:
        mask = mask[y_0:y_f, x_0:x_f]

    # Convert to grayscale
    img1_region_gray = cv2.cvtColor(img1_region, cv2.COLOR_BGR2GRAY)
    img2_region_gray = cv2.cvtColor(img2_region, cv2.COLOR_BGR2GRAY)

    # Equalize histograms to improve contrast
    img1_region_gray = cv2.equalizeHist(img1_region_gray)
    img2_region_gray = cv2.equalizeHist(img2_region_gray)

    # Track good features
    p1_region = cv2.goodFeaturesToTrack(img1_region_gray, mask=mask, **FEATURE_PARAMS)
    p2_region, st, _ = cv2.calcOpticalFlowPyrLK(  # pylint: disable=invalid-name
        img1_region_gray, img2_region_gray, p1_region, None, **LK_PARAMS
    )

    # Change origin and select tracked points
    p1_frame, p2_frame = p1_region + [x_0, y_0], p2_region + [x_0, y_0]
    p1_good, p2_good = p1_frame[st == 1], p2_frame[st == 1]

    # Cancel estimation if no good points were found or tracked
    if p1_good.size == 0:
        logging.error("No good points were found or sucessfully tracked.")
        return 3 * (0,), 3 * (0,), None

    # Estimate points and matrix
    p_good, affine_params = _estimate_matrix(
        p1_good, p2_good, quantile1=1.5, quantile2=1.5
    )
    p1_good, p2_good, p3_good = p_good

    # Mean square error
    err = _get_rmse(p2_good, p3_good)

    return p_good, affine_params, err
