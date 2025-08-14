# extract_landmarks.py
import csv
from recognizer.hand_position import analizar_posicion_manos

# Metadata fija
metadata = {
    "gloss": "book",
    "bbox": [385, 37, 885, 720],
    "fps": 25,
    "frame_end": -1,
    "frame_start": 1,
    "instance_id": 0,
    "signer_id": 118,
    "source": "aslbrick",
    "split": "train",
    "url": "http://aslbricks.org/New/ASL-Videos/book.mp4",
    "variation_id": 0,
    "video_id": "69241"
}

# 🚨 Aquí deberías tener un método que obtenga los landmarks.
# Por ahora voy a simular la estructura que recibe `analizar_posicion_manos`
# porque no tenemos la lectura real de landmarks desde la URL en Python puro.
# Si tu detección la hace el frontend con MediaPipe JS, aquí podrías recibir
# ese JSON desde un archivo o endpoint.

# Ejemplo de datos simulados
landmarks_data = {
    "pose": [{"y": 0.5}],  # landmark cabeza
    "hands": [
        [{"y": 0.4}],  # mano arriba
        [{"y": 0.6}]   # mano abajo
    ]
}

# Procesar
resultados = analizar_posicion_manos(landmarks_data)

# Guardar en CSV
csv_filename = "landmarks_metadata.csv"

# Abrir en modo append si quieres acumular datos
with open(csv_filename, mode="a", newline="", encoding="utf-8") as csvfile:
    fieldnames = list(metadata.keys()) + ["landmarks_result"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Escribir encabezado si el archivo está vacío
    if csvfile.tell() == 0:
        writer.writeheader()

    row = {**metadata, "landmarks_result": "; ".join(resultados)}
    writer.writerow(row)

print(f"✅ Datos guardados en {csv_filename}")