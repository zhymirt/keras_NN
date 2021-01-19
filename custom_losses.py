import tensorflow as tf
from tensorflow import keras


def discrim_wasserstein_loss_fn(y_true, y_pred):
    difference = y_true - y_pred # tf.subtract(y_true, y_pred)
    return -tf.reduce_sum(difference, axis=-1)

def gen_wasserstein_loss_fn(y_true, y_pred):
    return -tf.reduce_mean(y_pred, axis=-1)

class DiscriminatorWassersteinLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return discrim_wasserstein_loss_fn(y_true, y_pred)

class GeneratorWassersteinLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return gen_wasserstein_loss_fn(y_true, y_pred)
