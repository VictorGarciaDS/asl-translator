# tests/test_hand_detection.py

from preprocess.hand_detection import HandDetector
import cv2

def test_hand_detector_on_camera():
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Cámara no disponible"
    ret, frame = cap.read()
    assert ret, "No se pudo leer frame"
    
    results = detector.detect(frame)
    assert results is not None, "No se obtuvo resultado"
    
    cap.release()