import tensorflow as tf
from tensorflow import keras


def discrim_wasserstein_loss_fn(y_true, y_pred):
    difference = tf.abs(tf.reduce_mean(y_true) - tf.reduce_mean(y_pred)) # tf.subtract(y_true, y_pred)
    return difference
    difference = y_true - y_pred # tf.subtract(y_true, y_pred)
    return 1 - tf.reduce_sum(difference, axis=-1)
# def discrim_wasserstein_gradient_penalty_loss_fn(y_true, y_pred):
#     r = tf.random.uniform(shape=[1])
#     x_hat = tf.math.scalar_mul(r) + 

def gen_wasserstein_loss_fn(y_true, y_pred):
    # return tf.reduce_mean(y_pred, axis=-1)
    return -tf.reduce_mean(y_pred, axis=-1)

def wasserstein_loss_fn(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

class DiscriminatorWassersteinLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return wasserstein_loss_fn(y_true, y_pred)
        return discrim_wasserstein_loss_fn(y_true, y_pred)

class GeneratorWassersteinLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return wasserstein_loss_fn(y_true, y_pred)
        return gen_wasserstein_loss_fn(y_true, y_pred)
