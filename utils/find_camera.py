# utils/find_camera.py

import cv2

def find_working_camera_index(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    raise RuntimeError("❌ No se encontró ninguna cámara funcional.")
