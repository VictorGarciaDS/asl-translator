from flask import Flask, render_template, jsonify, request
import os
from recognizer.hand_position import analizar_posicion_manos

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/list-static-models')
def list_models():
    path = os.path.join(app.root_path, 'static', 'models')
    files = os.listdir(path)
    return jsonify(files)

# ✅ NUEVO ENDPOINT PARA RECIBIR LANDMARKS
@app.route("/api/landmarks", methods=["POST"])
def recibir_landmarks():
    data = request.get_json()
    resultados = analizar_posicion_manos(data)

    # Imprime en consola para debugging
    print("Datos recibidos del frame:")
    for key, value in data.items():
        print(f"{key}: {value}")
    
    return jsonify({
        "status": "ok",
        "resultados": resultados,   # opcional
        "data": data                # <-- aquí retornamos todos los landmarks
    })
if __name__ == "__main__":
    app.run(debug=True)