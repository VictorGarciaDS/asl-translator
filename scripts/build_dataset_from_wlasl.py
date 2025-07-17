# scripts/build_dataset_from_wlasl.py

import os
import json
import cv2
from tqdm import tqdm
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from preprocess.hand_detection import HandDetector
from recognizer.simple_classifier import HandGestureClassifier

VIDEO_DIR = "raw_videos"
JSON_PATH = "metadata/WLASL_v0.3.json"
OUTPUT_MODEL = "models/wlasl_svm_model.pkl"
MAX_FRAMES_PER_VIDEO = 30  # Puedes ajustar esto

# Carga el JSON
with open(JSON_PATH, "r") as f:
    wlasl_data = json.load(f)

detector = HandDetector()
classifier = HandGestureClassifier()  # No pasamos modelo porque vamos a entrenarlo

X, y = [], []

for entry in tqdm(wlasl_data, desc="Procesando señas"):
    gloss = entry["gloss"]  # La etiqueta
    for instance in entry["instances"]:
        video_id = instance["video_id"]
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened() and frame_count < MAX_FRAMES_PER_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break

            results = detector.detect(frame)
            if results.multi_hand_landmarks:
                emb = classifier.extract_landmarks(results.multi_hand_landmarks[0])
                if emb is not None:
                    X.append(emb)
                    y.append(gloss)
                    frame_count += 1
        cap.release()

print(f"\nTotal ejemplos recopilados: {len(X)}")

if X:
    classifier.fit(X, y)
    classifier.save(OUTPUT_MODEL)
    print(f"\n✅ Modelo guardado en: {OUTPUT_MODEL}")
else:
    print("\n⚠️ No se encontraron suficientes muestras.")
