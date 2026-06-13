import os, base64, cv2, numpy as np, json, hashlib, uuid, time
from datetime import datetime
from collections import defaultdict
from flask import (Flask, render_template, request, jsonify, session)
from functools import wraps
from utils.predictor import process_image

app = Flask(__name__)

# ── Security Config ───────────────────────────────────────────────────────────
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32))
app.config["MAX_CONTENT_LENGTH"]       = 10 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"]  = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"]   = False

# ── Rate Limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX    = 60
_rate_store = defaultdict(list)

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip  = request.remote_addr
        now = time.time()
        _rate_store[ip] = [t for t in _rate_store[ip] if now - t < RATE_LIMIT_WINDOW]
        if len(_rate_store[ip]) >= RATE_LIMIT_MAX:
            audit_log("RATE_LIMIT_EXCEEDED", ip=ip, endpoint=request.path)
            return jsonify({"error": "Too many requests. Please wait."}), 429
        _rate_store[ip].append(now)
        return f(*args, **kwargs)
    return decorated

# ── Directories ───────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

LOG_FILE   = "logs/activity_log.json"
AUDIT_FILE = "logs/audit_log.json"

# ── Magic byte validation ─────────────────────────────────────────────────────
MAGIC_BYTES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG":      "png",
    b"RIFF":         "webp",
}

def validate_image_bytes(data: bytes) -> bool:
    for magic in MAGIC_BYTES:
        if data[:len(magic)] == magic:
            return True
    return False

def allowed_extension(filename: str) -> bool:
    return ("." in filename and
            filename.rsplit(".", 1)[1].lower() in {"png","jpg","jpeg","webp"})

# ── Audit Logging ─────────────────────────────────────────────────────────────
def audit_log(event: str, **kwargs):
    logs = _load_json(AUDIT_FILE)
    prev_hash = logs[-1]["hash"] if logs else "GENESIS"
    entry = {
        "id":        str(uuid.uuid4()),
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event":     event,
        "ip":        kwargs.get("ip", request.remote_addr if request else "system"),
        "ua":        kwargs.get("ua", (request.user_agent.string[:120]
                                       if request and request.user_agent else "system")),
        **{k: v for k, v in kwargs.items() if k not in ("ip","ua")},
    }
    entry["hash"] = hashlib.sha256(
        (prev_hash + json.dumps(entry, sort_keys=True)).encode()
    ).hexdigest()
    logs.append(entry)
    _save_json(AUDIT_FILE, logs[-1000:])

# ── Activity Logging ──────────────────────────────────────────────────────────
def _load_json(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return []

def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def save_activity(entry):
    logs = _load_json(LOG_FILE)
    logs.append(entry)
    _save_json(LOG_FILE, logs[-500:])

def make_activity(source, faces, results):
    return {
        "id":          str(uuid.uuid4()),
        "timestamp":   datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source":      source,
        "faces_found": faces,
        "predictions": [
            {"label": r["label"], "confidence": round(r["confidence"]*100, 1)}
            for r in results
        ],
    }

# ── Security Headers ──────────────────────────────────────────────────────────
@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"]  = "nosniff"
    response.headers["X-Frame-Options"]         = "DENY"
    response.headers["X-XSS-Protection"]        = "1; mode=block"
    response.headers["Referrer-Policy"]         = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"]      = "geolocation=(), microphone=()"
    response.headers["Cache-Control"]           = "no-store, no-cache, must-revalidate"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "media-src 'self' blob:; "
        "frame-ancestors 'none';"
    )
    return response

# ── CSRF Token ────────────────────────────────────────────────────────────────
@app.before_request
def ensure_csrf():
    if "csrf_token" not in session:
        session["csrf_token"] = uuid.uuid4().hex

@app.context_processor
def inject_csrf():
    return {"csrf_token": session.get("csrf_token", "")}

# ── Image helper ──────────────────────────────────────────────────────────────
def bgr_to_base64(image_bgr):
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    audit_log("PAGE_VISIT", endpoint="/")
    return render_template("index.html")

@app.route("/webcam")
def webcam():
    audit_log("PAGE_VISIT", endpoint="/webcam")
    return render_template("webcam.html")

@app.route("/records")
def records():
    audit_log("PAGE_VISIT", endpoint="/records")
    return render_template("records.html")

@app.route("/predict", methods=["POST"])
@rate_limit
def predict():
    if not request.content_type or "multipart" not in request.content_type:
        audit_log("INVALID_CONTENT_TYPE", endpoint="/predict")
        return jsonify({"error": "Invalid request format"}), 400
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    if not file.filename or not allowed_extension(file.filename):
        audit_log("INVALID_FILE_TYPE", filename=str(file.filename)[:50])
        return jsonify({"error": "Invalid file type. Use PNG or JPEG."}), 400
    raw = file.read()
    if len(raw) > 10 * 1024 * 1024:
        return jsonify({"error": "File too large. Max 10MB."}), 413
    if not validate_image_bytes(raw):
        audit_log("MAGIC_BYTE_FAIL", filename=str(file.filename)[:50])
        return jsonify({"error": "File content does not match image type."}), 400
    file_bytes = np.frombuffer(raw, dtype=np.uint8)
    image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({"error": "Could not process image."}), 400
    h, w = image_bgr.shape[:2]
    if h > 4000 or w > 4000:
        image_bgr = cv2.resize(image_bgr, (min(w,4000), min(h,4000)))
    try:
        annotated, results = process_image(image_bgr)
    except Exception:
        audit_log("INFERENCE_ERROR", endpoint="/predict")
        return jsonify({"error": "Inference failed. Please try again."}), 500
    save_activity(make_activity("upload", len(results), results))
    audit_log("PREDICTION_OK", source="upload", faces=len(results))
    return jsonify({
        "annotated_image": bgr_to_base64(annotated),
        "faces_found":     len(results),
        "results":         results,
    })

@app.route("/predict_frame", methods=["POST"])
@rate_limit
def predict_frame():
    if not request.is_json:
        return jsonify({"error": "JSON required"}), 400
    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "No frame data"}), 400
    frame_str = data["frame"]
    if not isinstance(frame_str, str) or "," not in frame_str:
        return jsonify({"error": "Invalid frame format"}), 400
    header, encoded = frame_str.split(",", 1)
    if "image" not in header:
        audit_log("INVALID_FRAME_HEADER")
        return jsonify({"error": "Invalid frame header"}), 400
    if len(encoded) > 2 * 1024 * 1024:
        return jsonify({"error": "Frame too large"}), 413
    try:
        img_bytes = base64.b64decode(encoded, validate=True)
    except Exception:
        return jsonify({"error": "Invalid base64 data"}), 400
    nparr     = np.frombuffer(img_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({"error": "Could not decode frame"}), 400
    try:
        annotated, results = process_image(image_bgr)
    except Exception:
        return jsonify({"error": "Inference failed"}), 500
    if results:
        save_activity(make_activity("webcam", len(results), results))
    return jsonify({
        "annotated_frame": bgr_to_base64(annotated),
        "faces_found":     len(results),
        "results":         results,
    })

@app.route("/api/logs")
@rate_limit
def api_logs():
    logs = _load_json(LOG_FILE)
    logs.reverse()
    return jsonify(logs)

@app.route("/api/logs/clear", methods=["POST"])
@rate_limit
def clear_logs():
    _save_json(LOG_FILE, [])
    audit_log("LOGS_CLEARED")
    return jsonify({"success": True})

@app.route("/api/stats")
@rate_limit
def api_stats():
    logs     = _load_json(LOG_FILE)
    total    = len(logs)
    happy    = sum(1 for l in logs for p in l.get("predictions",[]) if p["label"]=="Happy")
    sad      = sum(1 for l in logs for p in l.get("predictions",[]) if p["label"]=="Sad")
    webcam_c = sum(1 for l in logs if l.get("source")=="webcam")
    upload_c = sum(1 for l in logs if l.get("source")=="upload")
    return jsonify({
        "total_sessions":   total,
        "happy_detections": happy,
        "sad_detections":   sad,
        "webcam_sessions":  webcam_c,
        "upload_sessions":  upload_c,
    })

@app.route("/api/audit")
@rate_limit
def api_audit():
    logs = _load_json(AUDIT_FILE)
    logs.reverse()
    return jsonify(logs[:200])

# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request"}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum 10MB."}), 413

@app.errorhandler(429)
def too_many(e):
    return jsonify({"error": "Too many requests. Please slow down."}), 429

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "An internal error occurred."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)