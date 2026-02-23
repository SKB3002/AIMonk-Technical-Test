# masked_bce.py
import tensorflow as tf

class MaskedBCE(tf.keras.losses.Loss):
    def __init__(self, pos_weights=None, name="masked_bce", reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE):
        super().__init__(name=name, reduction=reduction)
        self.pos_weights = pos_weights

    def call(self, y_true, y_pred):
        # y_true: (batch, 4) with -1 for NA
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)

        # Replace NA labels with 0 to avoid NaNs
        y_true_clean = tf.where(y_true == -1.0, tf.zeros_like(y_true), y_true)

        # Compute BCE with logits
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true_clean,
            logits=y_pred,
            pos_weight=self.pos_weights
        )

        loss = loss * mask
        return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-6)

    def get_config(self):
        config = super().get_config()
        config.update({
            "pos_weights": self.pos_weights.numpy().tolist() if self.pos_weights is not None else None
        })
        return config

    @classmethod
    def from_config(cls, config):
        pos_weights = config.pop("pos_weights", None)
        if pos_weights is not None:
            pos_weights = tf.constant(pos_weights, dtype=tf.float32)
        return cls(pos_weights=pos_weights, **config)
