import os, numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH  = "model/emotion_model.h5"
DATA_DIR    = "data/processed/test"
IMG_SIZE    = 48
BATCH_SIZE  = 64
CLASS_NAMES = ["happy", "sad"]

model = tf.keras.models.load_model(MODEL_PATH)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE,IMG_SIZE),
    color_mode="grayscale", class_mode="categorical",
    batch_size=BATCH_SIZE, shuffle=False, classes=CLASS_NAMES
)

preds       = model.predict(test_gen, verbose=1)
pred_labels = np.argmax(preds, axis=1)
true_labels = test_gen.classes

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=CLASS_NAMES))

cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png", dpi=150)
print("Saved: model/confusion_matrix.png")