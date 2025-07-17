# preprocess/hand_detection.py

import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandDetector:
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.7):
        self.hands = mp_hands.Hands(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence)
    
    def detect(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(image_rgb)

    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    def extract_hand_landmarks(self, results):
        """
        Extrae los landmarks de la primera mano detectada y los normaliza a un vector 1D.
        Retorna None si no hay mano detectada.
        """
        if not results.multi_hand_landmarks:
            return None
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        # Normalización simple (puedes mejorarla según convenga)
        landmarks = np.array(landmarks)
        landmarks = (landmarks - landmarks.min()) / (landmarks.max() - landmarks.min() + 1e-6)
        return landmarks