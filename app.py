# app.py

from flask import Flask, render_template, Response
import cv2
from preprocess.hand_detection import HandDetector
from capture.camera import get_camera_stream
from recognizer.simple_classifier import HandGestureClassifier
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Inicializa Flask
app = Flask(__name__, template_folder='templates')

# Cámara y detector
camera = get_camera_stream()
detector = HandDetector()

# Modelo de clasificación
model_path = "models/wlasl_svm_model.pkl"#"models/simple_svm_model.pkl"
classifier = HandGestureClassifier(model_path=model_path)

# Texto simulado (luego será dinámico)
recognized_text = "Esperando seña..."


# Generador de frames para el video
def gen_frames():
    global recognized_text
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            results = detector.detect(frame)
            detector.draw_landmarks(frame, results)

            # Extrae y normaliza landmarks de la mano
            landmarks = detector.extract_hand_landmarks(results)
            if landmarks is not None:
                recognized_text = classifier.predict(landmarks)

            # Codifica frame como JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Devuelve frame con mimetype correcto
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html', text=recognized_text)

# Ruta de video streaming
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ejecutar servidor
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)