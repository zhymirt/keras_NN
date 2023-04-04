""" Zhymir Thompson 2022
    Aggregate python file for scoring functions across packages."""
import tensorflow as tf
# import scipy
import numpy as np
import scipy.signal as signal

from scipy import fft, stats


# from sine_gan.py
def cross_correlate_mse(dataset, synth):
    # Only works on synth of 1
    # index of 0 lag when lengths are equal
    # lag_0 = synth[0].shape[0] - 1  # length of inner array minus one
    lag_0 = synth.shape[1] - 1
    dataset_corr = list()
    for data_point in dataset:
        avg = np.average([signal.correlate(data_point, synth_point)[lag_0] for synth_point in synth])
        dataset_corr.append(avg)
    dataset_corr = np.array(dataset_corr)
    # dataset_corr = np.array([signal.correlate(dataset[idx], synth[0])[lag_0] for idx in range(dataset.shape[0])])
    # print(dataset_corr.shape)
    # print(dataset_corr)
    # print(dataset_corr.max())
    return dataset_corr.max()


def auto_correlate_mse(dataset, synth):
    # Only works on synth of 1
    dataset_corr = np.array([])
    synth_corr = None
    pass

# from AF_gan.py


def get_auto_correlate_score(dataset, synth):
    """ Calculate and return auto correlate score."""
    # Only works for synthesized of size 1 for now
    def auto_correlate(arr):
        return signal.correlate(arr, arr)
    synth_auto_corr = np.apply_along_axis(auto_correlate, 1, synth)
    synth_auto_corr = np.array(
        [signal.correlate(
            synth[0, idx], synth[0, idx]) for idx in range(synth.shape[1])])
    dataset_auto_corr = np.array(
        [[signal.correlate(
            dataset[jdx, idx], dataset[jdx, idx]) for idx in range(
            dataset.shape[1])] for jdx in range(dataset.shape[0])])
    print(synth_auto_corr.shape)
    print(dataset_auto_corr.shape)
    mses = np.array(
        [np.average(np.square(
            dataset_auto_corr[data_point]
            - synth_auto_corr)) for data_point in range(dataset.shape[0])])
    print(mses.shape)
    print(mses)
    return np.min(mses)


def get_cross_correlate_score(dataset, synth):
    """ Compute and return cross correlate score."""
    correlates = []
    for d_sample, s_sample in zip(dataset, synth):
        correlate = signal.correlate(d_sample, s_sample)
        # print(correlate.shape)
        correlates.append(np.max(correlate))
    # correlate = signal.correlate(dataset, synth)
    correlates = np.asarray(correlates)
    print(correlates.shape)
    return np.average(np.asarray(correlates))


def get_fft_score(dataset, synth):
    """ Return FFT score."""
    # Get FFTs
    synth_fft, data_fft = fft.fft(np.asarray(synth)), fft.fft(np.asarray(dataset))
    min_diffs = list()
    for synth_obj in synth_fft:
        min_diff = 1e99
        for data_obj in data_fft:
            # diff = np.real(data_obj - synth_obj)
            # diff = np.square(diff)
            # diff = np.average(diff)
            diff = np.average(np.square(np.real(data_obj - synth_obj)))
            min_diff = min(min_diff, diff)
        min_diffs.append(min_diff)
    min_diffs = np.sqrt(np.real(min_diffs))
    return np.average(min_diffs)


@tf.function
def metric_fft_score(dataset, synth):
    """ Wrapper for get_fft_score."""
    return tf.cast(tf.py_function(
            get_fft_score, (dataset, synth), tf.complex64), tf.float32)


# from af_accel_GAN.py


def average_wasserstein(arr_1: np.ndarray, arr_2: np.ndarray) -> float:
    """ Calculate average wasserstein distance between two arrays."""
    arr_1, arr_2 = np.asarray(arr_1), np.asarray(arr_2)
    if arr_1.ndim == 1:
        return stats.wasserstein_distance(arr_1, arr_2)
    elif arr_1.ndim == 2:
        distances = [stats.wasserstein_distance(item_1, item_2)
                     for item_1, item_2 in zip(arr_1, arr_2)]
        return np.mean(distances)


@tf.function
def tf_avg_wasserstein(arr_1: np.ndarray, arr_2: np.ndarray) -> tf.float32:
    """ Wrap average_wasserstein function and return result."""
    return tf.py_function(average_wasserstein, (arr_1, arr_2), tf.float32)
