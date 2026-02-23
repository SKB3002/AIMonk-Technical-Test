#test_data_evaluator.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

def preprocess(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE)

logits = model.predict(val_ds)
probs = tf.sigmoid(logits).numpy()

# Clean NA labels
y_val_clean = np.where(y_val == -1, 0, y_val)
preds = (probs > 0.5).astype(int)

print(classification_report(
    y_val_clean.reshape(-1),
    preds.reshape(-1),
    zero_division=0
))
