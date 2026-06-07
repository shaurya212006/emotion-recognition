import os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from model.cnn_model import build_model

DATA_DIR     = "data/processed"
MODEL_PATH   = "model/emotion_model.h5"
HISTORY_PATH = "model/training_history.json"
IMG_SIZE     = 48
BATCH_SIZE   = 64
EPOCHS       = 60
CLASS_NAMES  = ["happy", "sad"]

train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=15,
    width_shift_range=0.1, height_shift_range=0.1,
    horizontal_flip=True, zoom_range=0.1, brightness_range=[0.8,1.2]
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR,"train"), target_size=(IMG_SIZE,IMG_SIZE),
    color_mode="grayscale", class_mode="categorical",
    batch_size=BATCH_SIZE, shuffle=True, classes=CLASS_NAMES
)
val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR,"val"), target_size=(IMG_SIZE,IMG_SIZE),
    color_mode="grayscale", class_mode="categorical",
    batch_size=BATCH_SIZE, shuffle=False, classes=CLASS_NAMES
)

model = build_model(input_shape=(IMG_SIZE,IMG_SIZE,1), num_classes=2)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

os.makedirs("model", exist_ok=True)
callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    CSVLogger("model/training_log.csv"),
]
happy_count = len(os.listdir(os.path.join(DATA_DIR,"train","happy")))
sad_count   = len(os.listdir(os.path.join(DATA_DIR,"train","sad")))
total       = happy_count + sad_count
print(f"Happy images: {happy_count}, Sad images: {sad_count}")

if happy_count == 0 or sad_count == 0:
    print("ERROR: No images found! Check data/processed/train/ folders.")
    exit()

class_weight = {0: total/(2*happy_count), 1: total/(2*sad_count)}

history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen,
                    callbacks=callbacks, class_weight=class_weight, verbose=1)

hist_dict = {k:[float(v) for v in vals] for k,vals in history.history.items()}
with open(HISTORY_PATH,"w") as f:
    json.dump(hist_dict, f, indent=2)

print(f"Best val accuracy: {max(hist_dict['val_accuracy']):.4f}")
print(f"Model saved to: {MODEL_PATH}")  