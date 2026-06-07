import os, base64, cv2, numpy as np
from flask import Flask, render_template, request, jsonify
from utils.predictor import process_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
os.makedirs("static/uploads", exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in {"png","jpg","jpeg","webp"}

def bgr_to_base64(image_bgr):
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({"error": "Could not decode image"}), 400
    annotated, results = process_image(image_bgr)
    return jsonify({"annotated_image": bgr_to_base64(annotated),
                    "faces_found": len(results), "results": results})

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    data = request.get_json(force=True)
    if not data or "frame" not in data:
        return jsonify({"error": "No frame"}), 400
    header, encoded = data["frame"].split(",", 1)
    img_bytes = base64.b64decode(encoded)
    nparr     = np.frombuffer(img_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    annotated, results = process_image(image_bgr)
    return jsonify({"annotated_frame": bgr_to_base64(annotated),
                    "faces_found": len(results), "results": results})

@app.route("/webcam")
def webcam():
    return render_template("webcam.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)