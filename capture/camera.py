# capture/camera.py

import cv2
from utils.find_camera import find_working_camera_index

def get_camera_stream():
    index = find_working_camera_index()
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir cámara en índice {index}")
    print(f"✅ Cámara abierta en índice {index}")
    return cap
