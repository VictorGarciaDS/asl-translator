# extract_landmarks.py
import cv2
import json
import csv
import requests

# URL del video (puedes cambiarla por cualquier otra en el futuro)
VIDEO_URL = "http://aslbricks.org/New/ASL-Videos/book.mp4"

# URL del backend (asegúrate que esté corriendo localmente o remoto)
BACKEND_URL = "http://127.0.0.1:5000/api/landmarks"

# Metadata fija (puede venir de un JSON externo)
VIDEO_METADATA = {
    "gloss": "book",
    "bbox": [385, 37, 885, 720],
    "fps": 25,
    "frame_end": -1,
    "frame_start": 1,
    "instance_id": 0,
    "signer_id": 118,
    "source": "aslbrick",
    "split": "train",
    "url": VIDEO_URL,
    "variation_id": 0,
    "video_id": "69241"
}

# CSV de salida
CSV_FILENAME = "landmarks_output.csv"

def enviar_frame_al_backend(frame_index):
    """
    Simula envío de frame al backend.
    Ahora simplemente le pedimos al backend que nos devuelva los landmarks.
    """
    # 🚨 Aquí podrías reemplazarlo por la detección real
    # por ahora el backend solo devuelve datos simulados/JSON vacío
    data = {
        "pose": [],   # opcional: si tienes detección real, enviar aquí
        "hands": [],
        "face": [],
    }
    try:
        response = requests.post(BACKEND_URL, json=data)
        if response.status_code == 200:
            json_resp = response.json()
            # Retornamos directamente los landmarks en "data"
            return json_resp.get("data", {})
        else:
            print(f"Error {response.status_code} al enviar frame {frame_index}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error conectando con backend en frame {frame_index}:", e)
        return None

def procesar_video():
    cap = cv2.VideoCapture(VIDEO_URL)
    frame_index = 0

    with open(CSV_FILENAME, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = list(VIDEO_METADATA.keys()) + ["frame_index", "landmarks"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            procesar_todo = (
                VIDEO_METADATA["frame_start"] == 1 and 
                VIDEO_METADATA["frame_end"] == -1
            )

            if procesar_todo or (
                VIDEO_METADATA["frame_start"] <= frame_index <= VIDEO_METADATA["frame_end"]
            ):
                if frame_index % 5 == 0:  # cada 5 frames
                    landmarks = enviar_frame_al_backend(frame_index)
                    if landmarks is not None:
                        writer.writerow({
                            **VIDEO_METADATA,
                            "frame_index": frame_index,
                            "landmarks": json.dumps(landmarks, ensure_ascii=False)
                        })

            frame_index += 1

    cap.release()
    print(f"✅ CSV guardado como {CSV_FILENAME}")

if __name__ == "__main__":
    procesar_video()