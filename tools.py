import cv2
import numpy as np
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
    return frame[cY-int(roi_heigh/2):cY+int(roi_heigh/2), cX-int(roi_width/2):cX+int(roi_width/2), :]


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
    for i in range(256):
        suma += ys[i]
        if suma/total > ant_ratio:
            threshold = i
            break

    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # return the reverted binary image
    return cv2.bitwise_not(thresh)


def get_ant_mask(window):
    return threshold_detector(window)