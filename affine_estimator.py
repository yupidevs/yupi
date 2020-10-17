import cv2
import numpy as np
import nudged
import math


# ShiTomasi corner detection
feature_params = dict(maxCorners = 20, 
                    qualityLevel = 0.5, 
                    minDistance = 5, 
                    blockSize = 100)

# Lucas Kanade optical flow
lk_params = dict(winSize = (50,50), 
               maxLevel = 15, 
               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 0.05))

           

def get_affine(img1, img2, show=True, debug=True):
    img1_gray =  cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)                    
    img2_gray =  cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(img1_gray, mask = None, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, p0, None, **lk_params)

     # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    center = (int(img1.shape[1]/2), int(img1.shape[0]/2))
    transformation = nudged.estimate(good_old - center, good_new - center)

    rows, cols, _ = img1.shape
    tx, ty = transformation.get_translation()
    theta = transformation.get_rotation()

    M = np.float32([[math.cos(theta), -math.sin(theta), tx], [math.sin(theta), math.cos(theta), ty]])

    img3 = cv2.warpAffine(img1, M, (cols,rows))

    if debug:
        print("features_used: {}".format(len(good_new)))
        print("translation: {}".format(transformation.get_translation()))
        print("rotation:  {}".format(transformation.get_rotation()))
        print("scale: {}".format(transformation.get_scale()))
        # # print("estimated error: "+str(nudged.estimate_error(transformation, good_old-center, good_new-center)))
        # print("epsilon: "+str(np.max(np.abs(good_new-good_old))))

    if show:
        for i in np.int0(good_old):
            x, y = i.ravel()
            cv2.circle(img1, (x,y), 3, (0,0,0), -1)

        for i in np.int0(good_new):
            x, y = i.ravel()
            cv2.circle(img2, (x,y), 3, (0,0,0), -1) 

        cv2.imshow('img1', img1)
        cv2.imshow('img2', img2)
        cv2.imshow('img3', img3)
        cv2.waitKey(-1)

