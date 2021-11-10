import tensorflow as tf


class LogImagesCallback(tf.keras.callbacks.Callback):
    def __init__(self, image_shape: tuple, num_images: int = 1):
        super().__init__(self)
        self.noise_vectors = tf.random.normal(shape=((num_images,) + image_shape))

    def on_epoch_end(self, epoch, logs=None):
        pass


class PrintLogsCallback(tf.keras.callbacks.Callback):
    """ Callback to print logs."""
    def on_epoch_begin(self, epoch, logs=None):
        print('Beginning logs: {}'.format(list(logs.keys())))

    def on_epoch_end(self, epoch, logs=None):
        print('Ending logs: {}'.format(list(logs.keys())))


class FFTCallback(tf.keras.callbacks.Callback):
    """ Callback for FFt Score."""
    def on_epoch_end(self, epoch, logs=None):
        print('FFT Score: {}'.format(logs['metric_fft_score']))