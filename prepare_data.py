# prepare_data.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

IMG_DIR = "/kaggle/input/datasets/suyashbhatkar2003/ai-monk-data-images/images"
LABEL_FILE = "/kaggle/input/datasets/suyashbhatkar2003/ai-monk-labels/labels.txt"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5

def load_labels(label_file):
    samples = []
    with open(label_file, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Skip header lines if any
    for line in lines:
        parts = line.split()
        if parts[0].lower().startswith("image"):
            img_name = parts[0]
            raw_labels = parts[1:]

            labels = []
            for x in raw_labels:
                if x == "NA":
                    labels.append(None)
                else:
                    labels.append(int(x))

            samples.append((img_name, labels))

    return samples

samples = load_labels(LABEL_FILE)

valid_samples = []
missing = []

for img_name, lbl in samples:
    img_path = os.path.join(IMG_DIR, img_name)
    if os.path.exists(img_path):
        valid_samples.append((img_name, lbl))
    else:
        missing.append(img_name)

print(f"Total samples in labels.txt: {len(samples)}")
print(f"Valid images found: {len(valid_samples)}")
print(f"Missing images: {len(missing)}")

samples = valid_samples

def compute_class_weights(samples):
    labels = np.array([
        [-1 if v is None else v for v in lbl] for _, lbl in samples
    ])
    weights = []
    for i in range(4):
        valid = labels[:, i] != -1
        pos = (labels[valid, i] == 1).sum()
        neg = (labels[valid, i] == 0).sum()
        w = neg / (pos + 1e-6)
        weights.append(w)
    return np.array(weights, dtype=np.float32)

POS_WEIGHTS = compute_class_weights(samples)

def preprocess_with_augmentations(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label
    
def preprocess(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label

img_paths = [os.path.join(IMG_DIR, s[0]) for s in samples]
labels = np.array([
    [-1 if v is None else v for v in s[1]] for s in samples
], dtype=np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    img_paths,
    labels,
    test_size=0.2,
    random_state=42
)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(512).map(preprocess_with_augmentations, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
