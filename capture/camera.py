# capture/camera.py

import cv2

def get_camera_stream(camera_index=0):
    return cv2.VideoCapture(camera_index)