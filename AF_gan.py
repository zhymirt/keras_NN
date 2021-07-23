import os
import cProfile
import tensorflow.experimental.numpy as tnp
import scipy.fft
import sklearn.preprocessing as preprocessing
import numpy as np
from scipy import signal
import tensorflow as tf
from matplotlib import pyplot as plt
from pyts.image.recurrence import RecurrencePlot
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from custom_losses import (DiscriminatorWassersteinLoss,
                           GeneratorWassersteinLoss, wasserstein_loss_fn, wasserstein_metric_fn)
from keras_data import data_to_dataset, plot_data, standardize
from keras_gan import WGAN, cWGAN, fft_callback
from keras_model_functions import get_recurrence, plot_recurrence
from model_architectures.AF_gan_architecture import (make_AF_discriminator,
                                                     make_AF_generator, make_AF_spectrogram_discriminator_1,
                                                     make_AF_spectrogram_generator_1)

keys = ['time', 'accel', 'intaccel', 'sg1', 'sg2', 'sg3']
save_paths = list(map(lambda x: 'time_' + x, keys[1:]))


def read_file(filename):
    """ Read file at given filename and return dictionary of lists composed of floats from file"""
    # data = {'time': [], 'accel': [], 'intaccel': [], 'sg1': [], 'sg2': [], 'sg3': []}
    data = {key: list() for key in keys}
    with open(filename, 'r') as tempFile:
        tempFile.readline()
        tempFile.readline()
        data_line = tempFile.readline()
        while data_line:
            split = data_line.split()
            for key, value in zip(keys, split):
                data[key].append(float(value))
            data_line = tempFile.readline()
    return data


def read_file_to_arrays(filename):
    time, data = list(), list()
    with open(filename, 'r') as tempFile:
        tempFile.readline()
        tempFile.readline()
        data_line = tempFile.readline()
        while data_line:
            split = data_line.split()
            time.append(float(split[0]))  # append time to time column
            data.append([float(value) for value in split[1:]])  # append data as 5-tuple
            data_line = tempFile.readline()
    return time, data

# def load_data(filename):
#     if not isinstance(filename, (str, list)):
#         print('No method to load this data')
#         return None, None
#     filename = [filename] if isinstance(filename, str) else filename # if string, make single element list
#     times, data_arrays = list(), list()
#     if filename[0].endswith('.txt'):
#         for name in filename:
#             time, data = read_file_to_arrays(name)
#             times.append(time)
#             data_arrays.append(data)
#     elif filename[0].endswith('.npy'):
#         data_arrays = [np.load(name) for name in filename]
#     else:
#         print("{} has unhandled extension".format(filename[0]))


def signal_dict_to_list(data):
    """ Take signal dictionary and convert to list, time excluded"""
    # signal_data = [ data[val] for val in keys[1:]] # as 5 rows of 5501
    indices = [idx for idx in range(len(data['time']))]
    signal_data = list(map(lambda x: [data[key][x] for key in keys[1:]], indices))
    return signal_data  # grab all but time list and make list


def make_dataset():
    pass


def plot_recurrence_diff(ref_data, synthetic_data, tmp_rp=None):
    # recurrence difference plot
    tmp_rp = RecurrencePlot() if tmp_rp is None else tmp_rp
    print('Ref num dimensions: {}, Synth num dimensions: {}'.format(ref_data.ndim, synthetic_data.ndim))
    if ref_data.ndim == 1:
        ref_data = np.expand_dims(ref_data, 0)
    if synthetic_data.ndim == 1:
        synthetic_data = np.expand_dims(synthetic_data, 0)
    for ref_signal, synthetic_signal in zip(ref_data, synthetic_data):
        temp_fig = plt.figure()
        ax = temp_fig.subplots(ncols=2, sharey=True)
        temp_orig = get_recurrence(ref_signal, tmp_rp)
        temp_synth = get_recurrence(synthetic_signal, tmp_rp)
        temp_fig.suptitle('Recurrence Plot Comparisons')
        # temp_diff = temp_orig - temp_synth
        ax[0].imshow(temp_orig, cmap='binary', origin='lower')
        ax[1].imshow(temp_synth, cmap='binary', origin='lower')
        # ax[2].imshow(temp_diff, cmap='binary', label='Difference btwn Recurrence Plots', origin='lower')
        ax[0].set_title('Reference Recurrence Plot')
        ax[1].set_title('Synthetic Recurrence Plot')
        # ax[2].set_title('Recurrence plot Difference')
        # for idx, arr in enumerate((temp_orig, temp_synth, temp_diff)):
        #     ax[idx].imshow(arr, cmap='binary')
        temp_fig.tight_layout()


def plot_correlations(ref_data, synthetic_data):
    temp_corr_fig = plt.figure()
    # ax = temp_corr_fig.subplots(ncols=ref_data.shape[0], nrows=2, sharey=True)
    print('Ref num dimensions: {}, Synth num dimensions: {}'.format(ref_data.ndim, synthetic_data.ndim))
    if ref_data.ndim == 1:
        ref_data = np.expand_dims(ref_data, 0)
    if synthetic_data.ndim == 1:
        synthetic_data = np.expand_dims(synthetic_data, 0)
    for idx, (ref_signal, synthetic_signal) in enumerate(zip(ref_data, synthetic_data)):
        temp_orig_auto = signal.correlate(ref_signal, ref_signal)
        temp_synth_cross = signal.correlate(ref_signal, synthetic_signal)
        temp_synth_auto = signal.correlate(synthetic_signal, synthetic_signal)
        plt.subplot(2, ref_data.shape[0], idx + 1)
        plt.plot(temp_orig_auto, label='Reference Autocorrelation')
        plt.plot(temp_synth_auto, '--', label='Synthetic Autocorrelation')
        plt.title('Signal Autocorrelations')
        plt.legend()

        plt.subplot(2, ref_data.shape[0], ref_data.shape[0] + idx + 1)
        plt.plot(temp_synth_cross)  # , label='Cross Correlation'
        plt.plot(temp_synth_cross)  # , label='Cross Correlation'
        plt.title('Signal Cross Correlation')


def plot_data(time, data, ref_data=None, show=False, save=True, save_path=''):
    print('Ref num dimensions: {}, Synth num dimensions: {}'.format(ref_data.ndim, data.ndim))
    if ref_data.ndim == 1:
        ref_data = np.expand_dims(ref_data, 0)
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    if np.ndim(data) < 2 or np.ndim(data) > 3:
        return
    if np.ndim(data) == 2:
        n_features, n_rows = data.shape[0], 1 if ref_data is None else 2
        temp_fig = plt.figure()
        # temp_ax = temp_fig.subplots(nrows=(1 if ref_data is None else 2), ncols=n_features, sharex=True)
        if ref_data is not None:
            for idx in range(n_features):
                plt.subplot(n_rows, n_features, idx + 1)
                plt.plot(time, data[idx])
                plt.title('Generated signal')
                plt.subplot(n_rows, n_features, n_features + idx + 1)
                plt.plot(time, ref_data[idx])
                plt.title('Reference Signal')
                # temp_ax[0, idx].plot(time, data[idx])
                # temp_ax[1, idx].plot(time, ref_data[idx])
        else:
            for idx in range(n_features):
                plt.subplot(n_rows, n_features, idx + 1)
                plt.plot(time, data[idx])
                plt.title('Generated signal')
                # temp_ax[idx].plot(time, data[idx])
    elif np.ndim(data) == 3:
        num_features,  num_rows = data.shape[1], 1 if ref_data is None else 2
        for sample_idx in range(data.shape[0]):
            temp_fig = plt.figure()
            # temp_ax = temp_fig.subplots(nrows=(1 if ref_data is None else 2), ncols=num_features, sharex=True)
            if ref_data is not None:
                for idx in range(num_features):
                    plt.subplot(num_rows, num_features, idx + 1)
                    plt.plot(time, data[idx])
                    plt.subplot(num_rows, num_features, num_features + idx + 1)
                    plt.plot(time, ref_data[idx])
                    # temp_ax[0, idx].plot(time, data[idx])
                    # temp_ax[1, idx].plot(time, ref_data[idx])
            else:
                for idx in range(num_features):
                    plt.subplot(num_rows, num_features, idx + 1)
                    plt.plot(time, data[idx])
                    # temp_ax[idx].plot(time, data[idx])
    if save and save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def generate_generic_training_data(time_frame, num_signals=1, frequencies=[1], amplitudes=[1], h_offsets=[0],
                                   v_offsets=[0]):
    # should be done in parallel to improve performance
    r_amplitudes = np.random.choice(amplitudes, size=num_signals)
    r_frequencies = np.random.choice(frequencies, size=num_signals)
    r_h_offsets = np.random.choice(h_offsets, size=num_signals)
    r_v_offsets = np.random.choice(v_offsets, size=num_signals)
    # end of parallel run
    signals = [(amplitude * np.sin(2 * np.pi * frequency * (time_frame + h_offset))) + v_offset for amplitude, frequency,
                                                                                              h_offset, v_offset in zip(
                                                                r_amplitudes, r_frequencies, r_h_offsets, r_v_offsets)]
    # signal = amplitude * np.sin(2 * np.pi * frequency * time)
    return signals


def get_auto_correlate_score(dataset, synth):
    # Only works for synthesized of size 1 for now
    synth_auto_corr = np.array([signal.correlate(synth[0, idx], synth[0, idx]) for idx in range(synth.shape[1])])
    dataset_auto_corr = np.array([[signal.correlate(dataset[jdx, idx], dataset[jdx, idx]) for idx in range(dataset.shape[1])] for jdx in range(dataset.shape[0])])
    print(synth_auto_corr.shape)
    print(dataset_auto_corr.shape)
    mses = np.array([np.average(np.square(dataset_auto_corr[data_point] - synth_auto_corr)) for data_point in range(dataset.shape[0])])
    print(mses.shape)
    print(mses)
    return np.min(mses)


def get_cross_correlate_score(dataset, synth):
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
    # Get ffts
    synth_fft, data_fft = scipy.fft.fft(np.array(synth)), scipy.fft.fft(np.array(dataset))
    min_diffs = list()
    for synth_obj in synth_fft:
        min_diff = 1e99
        for data_obj in data_fft:
            diff = np.real(data_obj - synth_obj)
            diff = np.square(diff)
            diff = np.average(diff)
            min_diff = min(min_diff, diff)
        min_diffs.append(min_diff)
    min_diffs = np.sqrt(np.real(min_diffs))
    return np.average(min_diffs)


@tf.function
def metric_fft_score(dataset, synth):
    return tf.cast(tf.py_function(get_fft_score, (dataset, synth), tf.complex64), tf.float32)


def normalize_data(data):
    norm_data = None
    if np.ndim(data) < 2 or np.ndim(data) > 3:
        return None
    elif np.ndim(data) == 3:
        norm_data = data.reshape((data.shape[1], -1))
    elif np.ndim(data) == 2:
        norm_data = data.reshape((data.shape[0], -1))
    norm_data, norm = preprocessing.normalize(norm_data, norm='l2', axis=1, return_norm=True)
    norm_data = norm_data.reshape(data.shape)
    return norm_data, norm


def denormalize_data(data, norms):
    norm_data = None
    if np.ndim(data) < 2 or np.ndim(data) > 3:
        return None
    elif np.ndim(data) == 3:
        norm_data = data.reshape((data.shape[1], -1))
    elif np.ndim(data) == 2:
        norm_data = data.reshape((data.shape[0], -1))
    resized = np.array([norm_data[idx, :] * norm_val for idx, norm_val in enumerate(norms)])
    resized = resized.reshape(data.shape)
    return resized


if __name__ == '__main__':
    data_type, batch_size, method = 'float32', 2, 'standard'
    # print(os.getcwd())
    # exit()
    time, benign_data = read_file_to_arrays('../signal_data/T04.txt')[0], [
        read_file_to_arrays(os.path.join('../signal_data', name))[1] for name in ['T04.txt',
                                                                                  'T04repeat.txt', 'T05.txt', 'T06.txt',
                                                                                  'T07.txt', 'T08.txt']]
    benign_data = np.array(benign_data[0:]) # Training on one example to narrow down issue
    # time, benign_data = np.load('../signal_data/time_np.npy'), np.concatenate(
    #     [[np.load(os.path.join('../signal_data', name + '_np.npy'))] for name in ['T04',
    #                                                                               'T04repeat', 'T05', 'T06', 'T07',
    #                                                                               'T08']])

    # Repeat data to increase data size
    # benign_data = benign_data.repeat(1e3, axis=0)
    print('Benign data shape: {}'.format(benign_data.shape))

    # # Generating generic data for few shot learning
    # init_training = generate_generic_training_data(time, num_signals=100*5, frequencies=range(1, 11),
    #                                                amplitudes=range(1, 1001), h_offsets=[0.01 * n for n in range(0, 100)])
    # init_training = np.reshape(init_training, (-1, 5501, 5))
    # print(init_training.shape)
    # generic_dataset = data_to_dataset(init_training, dtype=data_type, batch_size=batch_size)

    benign_data_transposed = np.transpose(benign_data, (0, 2, 1))
    data_size, num_data_types, latent_dimension = len(time), len(keys) - 1, 256
    # print(benign_data_transposed.shape)
    transformed, scalars = normalize_data(benign_data_transposed)
    resized = denormalize_data(transformed, scalars)
    # diff = np.abs(benign_data_transposed - resized)
    # print('Absolute Differences: min - {} max - {} total - {}'.format(diff.min(), diff.max(), diff.sum()))
    # print(benign_data_transposed[0, 1].max(), benign_data_transposed[0, 3].max())
    # print('Normalized shape: {}'.format(transformed.shape))
    # fig = plt.figure()
    # ax = fig.subplots(nrows=3, ncols=5) # 5 is magic number for number of plots
    # for idx in range(5):
    #     ax[0, idx].plot(time, benign_data_transposed[0, idx])
    #     ax[1, idx].plot(time, transformed[0, idx])
    #     ax[2, idx].plot(time, resized[0, idx])
    # plt.savefig('./results/raw_vs_normalized_plots.png')
    # plt.show()
    # exit()
    transformed = transformed.transpose((0, 2, 1))
    # Repeat data to increase data size
    transformed = transformed.repeat(2e3, axis=0)  # 1e4
    print('Repeated shape: '.format(transformed.shape))
    if method == 'spectrogram':
        fs = data_size / (time[-1] - time[0])
        # print(fs)
        # for idx in range(5):
        #     f, t, temp_sig = signal.spectrogram(benign_data_transposed[0, idx], fs, nperseg=500, noverlap=125)
        #     # print(temp_sig[2].shape)
        #     plt.figure()
        #     plt.pcolormesh(t, f, temp_sig, shading='gouraud')
        # plt.show()
        # spectrogram_scipy = []
        # for idx in range(benign_data_transposed.shape[0]):
        #     for jdx in range(benign_data_transposed.shape[1]):
        #         _, _, spec = signal.spectrogram(benign_data_transposed[idx][jdx][:], fs=fs)
        #         spectrogram_scipy.append(spec)
        # spectrogram_scipy = list(
        #     map(lambda x: signal.spectrogram(x, fs=data_size / (time[-1] - time[0]))[2], benign_data_transposed))
        # spectrogram_dataset = data_to_dataset(spectrogram_scipy, dtype=data_type, batch_size=batch_size)
        # # print(benign_data_transposed[0].shape)
        # # print(spectrogram_scipy[0].shape)
        # # plt.pcolormesh(spectrogram_scipy[1])
        # # plt.show()
        # # exit()
        # spectrogram_discriminator = make_AF_spectrogram_discriminator_1()
        # spectrogram_generator = make_AF_spectrogram_generator_1(latent_dimension)
        # spectrogram_wgan = WGAN(discriminator=spectrogram_discriminator, generator=spectrogram_generator,
        #                         latent_dim=latent_dimension)
        # spectrogram_wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
        #                          g_optimizer=keras.optimizers.Adam(learning_rate=0.0006)
        #                          )
        # spectrogram_wgan.set_train_epochs(4, 1)
        # print(spectrogram_scipy[0].shape)
        # print(spectrogram_discriminator.predict(tf.convert_to_tensor([spectrogram_scipy[0]])))
        # spectrogram_wgan.fit(spectrogram_dataset, batch_size=None, epochs=1024)
        # print(spectrogram_discriminator.predict(tf.convert_to_tensor([spectrogram_scipy[0]])))
        # plt.show()
        # rp = RecurrencePlot()
        # plot_recurrence(spectrogram_scipy[2], rp, show=True)
        # plot_recurrence(spectrogram_generator.predict(tf.random.normal(shape=(1, latent_dimension))), rp, show=True)
        # plt.pcolormesh(spectrogram_scipy[2], cmap='binary')
        # plt.show()
        # plt.pcolormesh(spectrogram_generator.predict(tf.random.normal(shape=(1, latent_dimension)))[0], cmap='binary')
        # plt.show()
        # exit()
    else:
        dataset = data_to_dataset(transformed, dtype=data_type, batch_size=batch_size, shuffle=True)
        # dataset = data_to_dataset(benign_data, dtype=data_type, batch_size=batch_size, shuffle=True)
        # create discriminator and generator

        discriminator = make_AF_discriminator(num_data_types, data_size)
        generator = make_AF_generator(latent_dimension, num_data_types, data_size)
        # generator attempts to produce even numbers, discriminator will tell if true or not
        wgan = WGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dimension)
        wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                     g_optimizer=keras.optimizers.Adam(learning_rate=0.0007),
                     metrics=[metric_fft_score, 'accuracy']
                     )
        wgan.set_train_epochs(4, 1)
        # # prefit with generic data for few shot learning
        # wgan.fit(generic_dataset, epochs=128, batch_size=batch_size)
        # Train model with real data
        early_stop = EarlyStopping(monitor='metric_fft_score', mode='min', min_delta=1e-2, verbose=1, patience=5)
        wgan.fit(dataset, epochs=0, batch_size=batch_size, callbacks=[fft_callback(), early_stop])
        # Saving models
        # generator.save('models/af_generator_full')
        # discriminator.save('models/af_discriminator_full')
        rp = RecurrencePlot()
        prediction = generator.predict(tf.random.normal(shape=(1, latent_dimension)))[0].transpose(1, 0)
        # recurrence difference plot
        plot_recurrence_diff(benign_data_transposed[0], prediction)
        # plt.show()
        plot_correlations(transformed[0].transpose(1, 0), prediction)
        plt.show()
        prediction = generator.predict(tf.random.normal(shape=(1, latent_dimension)))[0]
        print(get_cross_correlate_score(benign_data_transposed, generator.predict(tf.random.normal(shape=(1, latent_dimension))).transpose(0, 2, 1)))
        print(get_fft_score(transformed[0:128].transpose(0, 2, 1), generator.predict(tf.random.normal(shape=(64, latent_dimension))).transpose(0, 2, 1)))  # time,
        plot_data(time, np.transpose(np.array(prediction), (1, 0)), transformed.transpose(0, 2, 1)[0], show=True, save=False,
                  save_path='./results/AF_5_23_21_')
