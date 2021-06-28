import tensorflow as tf
from tensorflow import keras


def discrim_wasserstein_loss_fn(y_true, y_pred):
    difference = tf.reduce_mean(y_true - y_pred)
    return difference

def gen_wasserstein_loss_fn(y_true, y_pred):
    # return tf.reduce_mean(y_pred, axis=-1)
    return -tf.reduce_mean(y_pred, axis=-1)

def wasserstein_loss_fn(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def wasserstein_metric_fn(y_true, y_pred):
    return tf.reduce_mean(y_pred)

class DiscriminatorWassersteinLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return wasserstein_loss_fn(y_true, y_pred)
        return discrim_wasserstein_loss_fn(y_true, y_pred)

class GeneratorWassersteinLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return wasserstein_loss_fn(y_true, y_pred)
        return gen_wasserstein_loss_fn(y_true, y_pred)
