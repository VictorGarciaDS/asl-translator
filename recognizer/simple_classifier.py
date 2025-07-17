# recognizer/simple_classifier.py

import numpy as np
from sklearn.svm import SVC
import joblib

class HandGestureClassifier:
    def __init__(self, model_path=None):
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = SVC(kernel='linear', probability=True)
        self.label_map = {}

    def extract_landmarks(self, hand_landmarks):
        if not hand_landmarks:
            return None
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)

    def predict(self, landmarks_vector):
        # landmarks_vector: np.array  (por ejemplo shape (63,) con 21 puntos x 3 coords)
        if landmarks_vector is None:
            return "No hand detected"

        # reshaping si hace falta (1, n_features)
        landmarks_vector = landmarks_vector.reshape(1, -1)

        pred = self.model.predict(landmarks_vector)
        return pred[0]

    def fit(self, X, y):
        self.model.fit(X, y)
        classes = sorted(list(set(y)))
        self.label_map = {i: label for i, label in enumerate(classes)}
        # re-fit to relabel
        y_indexed = [list(self.label_map.values()).index(label) for label in y]
        self.model.fit(X, y_indexed)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)