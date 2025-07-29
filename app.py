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
    for r in resultados:
        print(r)  # Mostramos en terminal (servidor)
    '''
    print("Frente landmarks:", data.get("forehead"))
    print("Ceja izquierda landmarks:", data.get("left_eyebrow"))
    print("Ceja derecha landmarks:", data.get("right_eyebrow"))
    print("Sien izquierdo:", data.get("sien_izquierda"))
    print("Sien derecho:", data.get("sien_derecha"))
    print("Ojo izquierdo landmarks:", data.get("ojo_izquierdo"))
    print("Ojo derecho landmarks:", data.get("ojo_derecho"))
    print("Iris izquierdo:", data.get("iris_izquierdo"))
    print("Iris derecho:", data.get("iris_derecho"))
    print("Nariz izquierda:", data.get("nariz_izquierda"))
    print("Nariz derecha:", data.get("nariz_derecha"))
    print("Nariz baja:", data.get("nariz_baja"))
    print("Mejilla izquierda:", data.get("left_cheek"))
    print("Mejilla derecha:", data.get("right_cheek"))
    print("Boca landmarks:", data.get("boca"))
    print("Mentón landmarks:", data.get("menton"))
    print("Cuello landmarks:", data.get("neck"))
    print("Pose landmarks:", data.get("pose"))
    print("Manos landmarks:", data.get("hands"))
    '''
    return jsonify({"status": "ok", "resultados": resultados})

if __name__ == "__main__":
    app.run(debug=True)