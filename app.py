# app.py

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from preprocess.hand_detection import HandDetector
from recognizer.simple_classifier import HandGestureClassifier
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Inicializa detector y modelo
detector = HandDetector()
model_path = "models/wlasl_svm_model.pkl"  # puedes cambiarlo dinámicamente si deseas
classifier = HandGestureClassifier(model_path=model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_url = request.json['image']
        encoded_data = data_url.split(',')[1]
        img_bytes = base64.b64decode(encoded_data)
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        results = detector.detect(frame)
        landmarks = detector.extract_hand_landmarks(results)

        if landmarks is not None:
            prediction = classifier.predict(landmarks)
        else:
            prediction = "No se detectó una mano"

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)