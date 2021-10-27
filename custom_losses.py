import tensorflow as tf
from tensorflow import keras, reduce_mean
from tensorflow.keras.losses import Loss


def discrim_wasserstein_loss_fn(y_true, y_pred):
    """ Return loss for wasserstein discriminator(deprecated)."""
    difference = reduce_mean(y_true - y_pred)
    return difference


def gen_wasserstein_loss_fn(y_true, y_pred):
    """ Return loss for wasserstein generator(deprecated)."""
    # return tf.reduce_mean(y_pred, axis=-1)
    return -reduce_mean(y_pred, axis=-1)


def wasserstein_loss_fn(y_true, y_pred):
    """ Return wasserstein loss."""
    return reduce_mean(y_true * y_pred)


def wasserstein_metric_fn(y_true, y_pred):
    """ Return average wasserstein prediction."""
    return reduce_mean(y_pred)


def ebgan_loss_fn(y_true, y_pred):
    """ Return loss for EBGAN."""
    return reduce_mean(tf.maximum(tf.zeros(shape=y_true.shape), (y_true - y_pred)))


class DiscriminatorWassersteinLoss(Loss):
    """ Class for wasserstein discriminator."""
    def call(self, y_true, y_pred):
        return wasserstein_loss_fn(y_true, y_pred)
        return discrim_wasserstein_loss_fn(y_true, y_pred)


class GeneratorWassersteinLoss(Loss):
    """ Class for wasserstein generator."""
    def call(self, y_true, y_pred):
        return wasserstein_loss_fn(y_true, y_pred)
        return gen_wasserstein_loss_fn(y_true, y_pred)
