# masked_bce.py
import tensorflow as tf

class MaskedBCE(tf.keras.losses.Loss):
    def __init__(self, pos_weights, name="masked_bce"):
        super().__init__(name=name)
        self.pos_weights = tf.constant(pos_weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        y_true_clean = tf.where(y_true == -1.0, 0.0, y_true)

        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true_clean,
            logits=y_pred,
            pos_weight=self.pos_weights
        )
        return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
