# tests/test_recognizer.py

import numpy as np
from recognizer.simple_classifier import HandGestureClassifier

def test_classifier_prediction():
    classifier = HandGestureClassifier()
    X = [np.random.rand(63) for _ in range(10)]
    y = ["hola"] * 5 + ["gracias"] * 5
    classifier.fit(X, y)
    pred = classifier.model.predict([X[0]])
    assert pred[0] in [0, 1]
