from math import sqrt
from statistics import mean
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

def std_dev(vector):
    if vector is None or len(vector) == 0:
        return 0
    size = len(vector)
    avg = sum(vector)/size
    return sqrt(sum(list(map(lambda x: (x - avg)**2, vector)))/size)

def standardize(vector):
    avg = mean(vector)
    deviation = std_dev(vector)
    return list(map(lambda x: ((x - avg) / deviation) if deviation != 0 else 0, vector))

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

def load_model(path):
    model = tf.keras.models.load_model(path)
    return model

def load_data(path):
    data = np.load(path)
    return data

# def save_data(path, data, compress=False):
#     if compress:
#         np.savez()
#     else:
#         np.save(path, data)
