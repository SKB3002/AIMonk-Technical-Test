#model_finetune.py
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False, 
    weights='imagenet' 
)

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(256, activation="relu")(x)
outputs = tf.keras.layers.Dense(4)(x)  # logits

model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

# 2) Compile model 
loss_fn = MaskedBCE(POS_WEIGHTS)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=loss_fn
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,       
    callbacks=callbacks
)

model.save("multilabel_resnet50_tf.h5")

losses = history.history["loss"]

plt.plot(losses)
plt.xlabel("iteration_number")
plt.ylabel("training_loss")
plt.title("Aimonk_multilabel_problem")
plt.savefig("training_loss_curve.png")
plt.show()
