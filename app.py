from flask import Flask, render_template, jsonify
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

if __name__ == "__main__":
    app.run(debug=True)