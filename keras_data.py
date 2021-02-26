import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

def plot_data(x_values, y_values, trend_data=None, show=False, save=True, save_path=''):
    plt.plot(x_values, y_values)
    if trend_data is not None and len(trend_data) > 0:
        plt.plot(x_values, trend_data)
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def data_to_dataset(data, dtype='float32', batch_size=64, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(tf.cast(data, dtype=dtype))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset
