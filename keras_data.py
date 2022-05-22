import numpy as np

from datetime import date
from math import sqrt
from statistics import mean
from numpy import ndarray
from scipy.signal import periodogram


# TODO split these for modularity with imports


def std_dev(vector):
    """ Return standard deviation of a vector."""
    if vector is None or len(vector) == 0:
        return 0  # TODO raise error here
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
    average = np.mean(vector)  # precalculate mean for speed
    # number of points minus 1 * the standard deviation squared
    denominator = (len(vector) - 1) * np.std(vector) ** 3
    return np.sum((vector - average) ** 3) / denominator


def kurtosis(vector: ndarray) -> float:
    """ Return kurtosis of vector."""
    average = np.mean(vector)  # precalculate mean for speed
    # number of points minus 1 * the standard deviation squared
    denominator = (len(vector) - 1) * np.std(vector) ** 4
    return np.sum((vector - average) ** 4) / denominator


def crest_factor(vector: ndarray) -> float:
    """ Return crest factor of vector."""
    return np.max(vector) / rms(vector)


def shape_factor(vector: ndarray) -> float:
    """ Return shape factor of vector."""
    return rms(vector) / abs_mean(vector)


def impulse_factor(vector: ndarray) -> float:
    """ Return impulse factor of vector."""
    return np.max(vector) / abs_mean(vector)


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


def load_data(path):
    """ Load and return numpy data at given path."""
    data = np.load(path)
    return data


def get_date_string():
    """ Return date string for today's date in year-month-day format."""
    return str(date.today())

# def save_data(path, data, compress=False):
#     if compress:
#         np.savez()
#     else:
#         np.save(path, data)

# if __name__=='__main__':
#     print(get_date_string())
