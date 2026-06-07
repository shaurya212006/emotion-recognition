import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "model", "emotion_model.h5")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CLASS_NAMES  = ["Happy", "Sad"]
IMG_SIZE     = 48

_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train.py first."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def detect_faces(gray_frame):
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces   = cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return faces if len(faces) else []


def preprocess_face(gray_frame, x, y, w, h):
    face = gray_frame[y:y + h, x:x + w]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face


def predict_emotion(face_tensor):
    model  = get_model()
    probs  = model.predict(face_tensor, verbose=0)[0]
    idx    = int(np.argmax(probs))
    label  = CLASS_NAMES[idx]
    conf   = float(probs[idx])
    probs_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return label, conf, probs_dict


def process_image(image_bgr):
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces   = detect_faces(gray)
    results = []

    for (x, y, w, h) in faces:
        face_tensor        = preprocess_face(gray, x, y, w, h)
        label, conf, probs = predict_emotion(face_tensor)
        results.append(dict(x=int(x), y=int(y), w=int(w), h=int(h),
                            label=label, confidence=conf, probs=probs))

        color = (50, 205, 50) if label == "Happy" else (60, 100, 220)
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)

        text = f"{label}  {conf * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image_bgr, (x, y - th - 10), (x + tw + 8, y), color, -1)
        cv2.putText(image_bgr, text, (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return image_bgr, results