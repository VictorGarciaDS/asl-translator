# app.py
from flask import Flask, send_from_directory, request, jsonify

app = Flask(__name__, static_folder="public")

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Ejemplo: sólo contamos cuántos landmarks hay
    num_pose = len(data.get("pose", []))
    num_face = len(data.get("face", []))
    num_hands = sum(len(hand) for hand in data.get("hands", []))

    # Aquí es donde más adelante puedes cargar tu modelo y hacer predicciones
    return jsonify({
        "status": "ok",
        "num_pose_landmarks": num_pose,
        "num_face_landmarks": num_face,
        "num_hand_landmarks": num_hands,
        "message": f"Pose: {num_pose}, Face: {num_face}, Hands: {num_hands}"
    })