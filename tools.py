import cv2
from settings import frame_diff_threshold

def frame_diff(frame1, frame2):
    # obtain the frame difference
    diff = cv2.absdiff(frame1, frame2)

    # convert image to grayscale image
    gray_image = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(gray_image, frame_diff_threshold, 255, cv2.THRESH_BINARY)

    # Calculate moments
    M = cv2.moments(thresh)  

    # Checks if something was over the threshold
    if M["m00"] != 0:
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        print('[ERROR] Nothing was over threshold\n Try decreasing the value of settings.frame_diff_threshold')
        return

    # put text and highlight the center
    cv2.circle(frame2, (cX, cY), 5, (255, 255, 255), -1)

    return frame2