import cv2
import os
import numpy as np
from tools import frame_diff
from settings import data_folder, data_file

cap = cv2.VideoCapture(os.path.join(os.getcwd(), data_folder, data_file))
previous_frame = None
while True:
    ret, frame = cap.read()    
    if ret:
        if not isinstance(previous_frame, np.ndarray):
            previous_frame = frame.copy()
            continue
        fd = frame_diff(previous_frame, frame)
        cv2.imshow('a', fd)
        cv2.waitKey(10)

        previous_frame = frame.copy()      
    else:
        print("[INFO] No more frames to process.")
        break

