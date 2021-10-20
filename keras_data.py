import numpy as np
import scipy as sp
import tensorflow as tf

from datetime import date
from math import sqrt
from statistics import mean
from numpy import ndarray
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import periodogram

from matplotlib import pyplot as plt


def std_dev(vector):
    """ Return standard deviation of a vector."""
    if vector is None or len(vector) == 0:
        return 0
    size = len(vector)
    avg = sum(vector)/size
    return sqrt(sum(list(map(lambda x: (x - avg)**2, vector)))/size)


def standardize(vector):
    """ Return vector standardized."""
    avg = mean(vector)
    deviation = std_dev(vector)
    return list(map(lambda x: ((x - avg)
                               / deviation) if deviation != 0 else 0, vector))


def abs_mean(vector: ndarray) -> float:
    """ Return absolute mean of vector."""
    return np.mean(np.abs(vector))


def rms(vector: ndarray) -> float:
    """ Return root mean square of vector."""
    return np.sqrt(np.mean(vector ** 2))


def skewness(vector: ndarray) -> float:
    """ Return skewness of vector."""
    mean = np.mean(vector)  # precalculate mean for speed
    # number of points minus 1 * the standard deviation squared
    denom = (len(vector) - 1) * np.std(vector) ** 3
    return np.sum((vector - mean) ** 3) / denom


def kurtosis(vector: ndarray) -> float:
    """ Return kurtosis of vector."""
    mean = np.mean(vector) # precalculate mean for speed
    # number of points minus 1 * the standard deviation squared
    denom = (len(vector) - 1) * np.std(vector) ** 4
    return np.sum((vector - mean) ** 4) / denom


def crest_factor(vector: ndarray) -> float:
    """ Return crest factor of vector."""
    return np.max(vector) / rms(vector)


def shape_factor(vector: ndarray) -> float:
    """ Return shape factor of vector."""
    return rms(vector) / abs_mean(vector)


def impulse_factor(vector: ndarray) -> float:
    """ Return impulse factor of vector."""
    return np.max(vector) / abs_mean(vector)


def get_fft(vector: ndarray) -> ndarray:
    """ Wrapper for FFT function."""
    return fft(vector)


def get_fft_freq(vector: ndarray, time_step: float) -> ndarray:
    """ Wrapper for FFT frequencies."""
    return fftfreq(vector.size, time_step)


def frequency_power_sum(vector: ndarray) -> ndarray:
    """ Returns sum of power over frequencies."""
    return sum(get_fft(vector))


def frequency_center(vector: ndarray, time_step: float) -> float:
    """ Return frequency center of vector."""
    frequencies, power = periodogram(vector, 1/time_step)
    numerator = np.sum(power * frequencies)
    denominator = np.sum(power)
    return numerator / denominator


def root_mean_square_frequency(vector: ndarray, time_step: float) -> float:
    """ Return root mean square frequency."""
    frequencies, power_spectrum = periodogram(vector, 1/time_step)
    return np.sqrt(np.sum(frequencies**2 * power_spectrum)
                   / np.sum(power_spectrum))


def root_variance_frequency(vector: ndarray, time_step: float) -> float:
    """ Return root variance frequency."""
    frequencies, power_spectrum = periodogram(vector, 1 / time_step)
    freq_center = calc_frequency_center(power_spectrum, frequencies)
    return np.sqrt(np.sum(((frequencies - freq_center) ** 2) * power_spectrum)
                   / np.sum(power_spectrum))


def calc_frequency_center(
        power_spectrum: ndarray, frequencies: ndarray) -> float:
    """ Return frequency center of vector."""
    return np.sum(power_spectrum * frequencies) / np.sum(power_spectrum)
    # return np.mean(fft)


def calc_root_mean_square_frequency(
        power_spectrum: ndarray, frequencies: ndarray) -> float:
    """ Return root mean square frequency."""
    return np.sqrt(np.sum(power_spectrum * frequencies ** 2)
                   / np.sum(power_spectrum))
    # return np.mean(np.abs(np.asarray(fft)) ** 2)


def calc_root_variance_frequency(
        power_spectrum: ndarray, frequencies: ndarray) -> float:
    """ Return root variance frequency."""
    freq_center = calc_frequency_center(power_spectrum, frequencies)
    return np.sqrt(np.sum(((frequencies - freq_center) ** 2) * power_spectrum)
                   / np.sum(power_spectrum))
    # fft = sp.fft.fft(vector)
    # mean = np.mean(fft)
    # return np.sqrt(np.mean((fft - mean) ** 2))


def plot_data(
        x_values, y_values, trend_data=None, show=False,
        save=True, save_path=''):
    """ Plot (x, y) pairs, plot secondary trendline if provided."""
    plt.plot(x_values, y_values)
    if trend_data is not None and len(trend_data) > 0:
        plt.plot(x_values, trend_data)
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def data_to_dataset(data, dtype='float32', batch_size=64, shuffle=True):
    """ Return dataset given numpy data."""
    dataset = tf.data.Dataset.from_tensor_slices(tf.cast(data, dtype=dtype))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset


def load_model(path):
    """ Load and return model at given path."""
    model = tf.keras.models.load_model(path)
    return model


def load_data(path):
    """ Load and return numpy data at given path."""
    data = np.load(path)
    return data


def get_date_string():
    """ Return date string for today's date."""
    return date.today()

# def save_data(path, data, compress=False):
#     if compress:
#         np.savez()
#     else:
#         np.save(path, data)

# if __name__=='__main__':
#     print(get_date_string())
