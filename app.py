from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Aquí iría la lógica real del backend usando los landmarks
    print("Datos recibidos:", data)
    return jsonify({'message': 'Landmarks recibidos correctamente', 'prediction': 'TODO'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)