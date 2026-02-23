# inference.py
import tensorflow as tf
import numpy as np
from PIL import Image
from masked_bce import MaskedBCE

IMG_SIZE = 224
ATTRS = ["Attr1", "Attr2", "Attr3", "Attr4"]

model = tf.keras.models.load_model(
    "model.keras",  # update path if needed
    custom_objects={"MaskedBCE": MaskedBCE}
)

def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return np.expand_dims(img, 0)

def predict(image_path, threshold=0.5):
    logits = model(preprocess_image(image_path))
    probs = tf.sigmoid(logits)[0].numpy()
    present = [ATTRS[i] for i, p in enumerate(probs) if p > threshold]
    print("Attributes present:", present)
    print("Probabilities:", probs)

if __name__ == "__main__":
    predict("path/to/test_image.jpg")
