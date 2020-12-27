import cv2
import numpy as np
import nudged
import math


# ShiTomasi corner detection
feature_params = dict(maxCorners = 20, 
                    qualityLevel = 0.6, 
                    minDistance = 30, 
                    blockSize = 100)

# Lucas Kanade optical flow
lk_params = dict(winSize = (50,50), 
               maxLevel = 15, 
               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 0.05))

           

def get_affine(img1, img2, regions, show=True, debug=True):
    errors = []
    retval = []
    for x0, xf, y0, yf  in regions:
        i1, i2 = img1[x0:xf, y0:yf, :], img2[x0:xf, y0:yf, :]
        img1_gray =  cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)                    
        img2_gray =  cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(img1_gray, mask = None, **feature_params)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, p0, None, **lk_params)

        # NOTA: p0 y P1 están con los ejes invertidos

         # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        center = (int(i1.shape[1]/2), int(i1.shape[0]/2))
        transformation = nudged.estimate(good_old - center, good_new - center)

        rows, cols, _ = i1.shape
        tx, ty = transformation.get_translation()
        theta = transformation.get_rotation()
        scale = transformation.get_scale()

        if abs(scale - 1) > 0.001:
            print(scale)
            errors.append(abs(scale - 1))
            retval.append((tx, ty, theta, scale,(x0, xf, y0, yf)))
            continue 

        if debug:
            print("features_used: {}".format(len(good_new)))
            print("translation: {} {}".format(tx, ty))
            print("rotation:  {}".format(theta))
            print("scale: {}".format(scale))
            # # print("estimated error: "+str(nudged.estimate_error(transformation, good_old-center, good_new-center)))
            # print("epsilon: "+str(np.max(np.abs(good_new-good_old))))

        # Fix once the affine is fixed
        if show:
            M = np.float32([[math.cos(theta), -math.sin(theta), tx], [math.sin(theta), math.cos(theta), ty]])
            i1 = i1.copy()
            i2 = i2.copy()
            i3 = cv2.warpAffine(i1, M, (cols,rows))

            for i in np.int0(good_old):
                x, y = i.ravel()
                cv2.circle(i1, (x,y), 3, (0,0,0), -1)

            for i in np.int0(good_new):
                x, y = i.ravel()
                cv2.circle(i2, (x,y), 3, (0,0,0), -1) 

            cv2.imshow('img1', i1)
            cv2.imshow('img2', i2)
            cv2.imshow('img3', i3)
            cv2.waitKey(-1)

        return tx, ty, theta, scale, (x0, xf, y0, yf)

    best = errors.index(min(errors))
    return retval[best]