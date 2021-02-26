import os
import cv2
import json
import numpy as np
import settings as sett


# Algorithm 

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


# ROI




def update_roi_center(img, prev_cXY):
    prev_cX, prev_cY = prev_cXY

    # get the centroid refered to the roi
    cX_roi, cY_roi = get_centroid(img)

    # get the centroid refered to the full image
    cX = prev_cX - int(sett.roi_width/2) + cX_roi
    cY = prev_cY - int(sett.roi_heigh/2) + cY_roi
    return cX, cY




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
    for x0, xf, y0, yf in roi_array:
        mask[y0:yf, x0:xf] = 0

    return mask

# Viewing options 




def resize_frame(frame, scale=0.5):
    h, w = frame.shape[:2]
    w_, h_ = int(scale * w), int(scale * h)
    short_frame = cv2.resize(frame, (w_, h_), interpolation=cv2.INTER_AREA)
    return short_frame


# def show_frame(frame, cXY=None, region=None, features=None, frame_numb=None,
#         scale=sett.resize_factor, win_name=sett.video_file[:-4], mask=None):
#     frame = draw_frame(frame, cXY, region, features, frame_numb, mask)
#     frame = resize_frame(frame, scale)
#     cv2.imshow(win_name, frame)
#     return


