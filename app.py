from flask import Flask, render_template, jsonify, request
import os

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
    print("Landmarks recibidos")
    print("Timestamp:", data.get("timestamp"))
    '''
    print("Frente landmarks:", data.get("forehead"))
    print("Cejas landmarks:", data.get("cejas"))
    print("Ojos landmarks:", data.get("ojos"))
    print("Iris landmarks:", data.get("iris"))
    print("Sienes landmarks:", data.get("temples"))
    print("Nariz landmarks:", data.get("nariz"))
    print("Boca landmarks:", data.get("boca"))
    print("Mejillas landmarks:", data.get("mejillas"))
    print("Mentón landmarks:", data.get("menton"))
    print("Pose landmarks:", data.get("pose"))
    print("Manos landmarks:", data.get("hands"))
    print("Cuello landmarks:", data.get("neck"))
    '''
    print("-" * 40)
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)