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

from custom_losses import (DiscriminatorWassersteinLoss,
                           GeneratorWassersteinLoss, wasserstein_loss_fn, wasserstein_metric_fn)
from keras_data import data_to_dataset, plot_data, standardize
from keras_gan import WGAN, cWGAN
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


def signal_dict_to_list(data):
    """ Take signal dictionary and convert to list, time excluded"""
    # signal_data = [ data[val] for val in keys[1:]] # as 5 rows of 5501
    indices = [idx for idx in range(len(data['time']))]
    signal_data = list(map(lambda x: [data[key][x] for key in keys[1:]], indices))
    return signal_data  # grab all but time list and make list


# def convert_to_time_data

def plot_data(time, data, ref_data=None, show=False, save=True, save_path=''):
    if np.ndim(data) < 2 or np.ndim(data) > 3:
        return
    if np.ndim(data) == 2:
        n_features = data.shape[0]
        temp_fig = plt.figure()
        temp_ax = temp_fig.subplots(nrows=(1 if ref_data is None else 2), ncols=n_features, sharex=True)
        if ref_data is not None:
            for idx in range(n_features):
                temp_ax[0, idx].plot(time, data[idx])
                temp_ax[1, idx].plot(time, ref_data[idx])
        else:
            for idx in range(n_features):
                temp_ax[idx].plot(time, data[idx])
    elif np.ndim(data) == 3:
        num_features = data.shape[1]
        for sample_idx in range(data.shape[0]):
            temp_fig = plt.figure()
            temp_ax = temp_fig.subplots(nrows=(1 if ref_data is None else 2), ncols=num_features, sharex=True)
            if ref_data is not None:
                for idx in range(num_features):
                    temp_ax[0, idx].plot(time, data[idx])
                    temp_ax[1, idx].plot(time, ref_data[idx])
            else:
                for idx in range(num_features):
                    temp_ax[idx].plot(time, data[idx])
    if save and save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    # for idx in range(len(save_paths)):
    #     y_values = list_y_values[idx]
    #     print(y_values)
    #     print(y_values.shape)
    #     plt.plot(x_values, y_values)
    #     plt.title(label=save_path.split('/')[-1] + save_paths[idx])
    #     if save and save_path:
    #         plt.savefig(save_path + save_paths[idx])
    #     if show:
    #         plt.show()
    #     plt.close()


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
    pass


def get_fft_score(dataset, synth): # time
    # N = len(time)
    # T = (time[-1] - time[0]) / N
    # dataset, synth = dataset.astype('float64'), synth.astype('float64')
    # synth_fft, data_fft = list(), list()
    # fft_fn = lambda x: scipy.fft.fft(x)
    # synth_fft = np.apply_along_axis(fft_fn, 2, synth) # np_fft
    # data_fft = np.apply_along_axis(fft_fn, 2, dataset) # dataset_fft
    synth_fft, data_fft = scipy.fft.fft(synth), scipy.fft.fft(dataset)
    # Get ffts
    # for synth_obj in range(synth.shape[0]):
    #     ffts = [scipy.fft.fft(synth[synth_obj, feature]) for feature in range(synth.shape[1])]
    #     synth_fft.append(ffts)
    # synth_fft = np.array(synth_fft)  # , dtype='float64'
    # print((np_fft - synth_fft).sum())
    # for data_obj in range(dataset.shape[0]):
    #     ffts = [scipy.fft.fft(dataset[data_obj, feature]) for feature in range(dataset.shape[1])]
    #     data_fft.append(ffts)
    # data_fft = np.array(data_fft)  # , dtype='float64'
    # print((dataset_fft - data_fft).sum())
    # print(synth_fft.shape)
    # print(data_fft.shape)
    min_diffs = list()
    expanded = np.repeat([synth_fft[0]], data_fft.shape[0], 0)
    # print('Expanded shape: {}'.format(expanded.shape))
    # min_diff_fn = lambda x: np.subtract(data_fft, np.repeat([x], data_fft.shape[0], axis=0))
    # np_min_diff = np.apply_along_axis(min_diff_fn, 2, synth_fft)
    for synth_obj in synth_fft:
        min_diff, diff = 1e99, None
        for data_obj in data_fft:
            diff = data_obj - synth_obj
            # print('Difference: {}'.format(diff))
            diff = np.square(diff)
            # print('Squared: {}'.format((diff < 0.0).any()))
            # print('Squared shape: {}'.format(diff.shape))
            diff = np.sum(diff, axis=1)
            # print('Summed shape: {}'.format(diff.shape))
            diff = np.average(np.sqrt(diff))
            print('Average: {}'.format(diff))
            # diff = np.average(np.square(data_fft[data_obj] - synth_fft))
            # print('Total squared difference: {}, Current minimum: {}'.format(diff, min_diff))
            min_diff = min(min_diff, diff)
        min_diffs.append(min_diff)
    min_diffs = np.array(min_diffs)
    # min_diffs = np.sqrt(np.array(min_diffs))
    # print(min_diffs.shape)
    # print(min_diffs)
    return np.average(min_diffs)


def metric_fft_score(dataset, synth):
    # tf.config.run_functions_eagerly(True)
    # return get_fft_score(dataset, synth)
    synth_fft, dataset_fft = tf.signal.fft(tf.cast(synth, tf.complex64)), tf.signal.fft(tf.cast(dataset, tf.complex64))
    # expanded = np.repeat([synth_fft[0]], data_fft.shape[0], 0)
    # print('Expanded shape: {}'.format(expanded.shape))
    # # min_diff_fn = lambda x: np.subtract(data_fft, np.repeat([x], data_fft.shape[0], axis=0))
    # # np_min_diff = np.apply_along_axis(min_diff_fn, 2, synth_fft)
    min_diffs = list()
    for synth_obj in synth_fft:
        min_diff, diff = 1e99, None
        for data_obj in dataset_fft:
            diff = data_obj - synth_obj
            # print('Difference: {}'.format(diff))
            diff = np.square(diff)
            print('Squared: {}'.format((diff < 0.0).any()))
            diff = np.sum(diff)
            print('Sum: {}'.format(diff))
            # diff = np.average(np.square(data_fft[data_obj] - synth_fft))
            print('Total squared difference: {}, Current minimum: {}'.format(diff, min_diff))
            min_diff = min(min_diff, diff)
        min_diffs.append(min_diff)
    min_diffs = np.sqrt(np.array(min_diffs))
    # print(min_diffs.shape)
    print(min_diffs)
    return np.average(min_diffs)

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


# class AFGANMetric(keras.metrics.Metric):
#     def __init__(self, name='cross_correlation_rmse', sampling_time=1, **kwargs):
#         super(AFGANMetric, self).__init__(name=name, **kwargs)
#         self.fft_score = 1e99
#         self.sampling_time = sampling_time
#
#     def update_state(self, *args, **kwargs):
#         pass
#
#     def result(self):
#         return self.fft_score
#
#     def reset_states(self):
#         self.fft_score = 1e99

class AFGAN(WGAN):

    # def compile(self, d_optimizer, g_optimizer, d_loss_fn=wasserstein_loss_fn, g_loss_fn=wasserstein_loss_fn):
    #     super(WGAN, self).compile()
    #     self.d_optimizer = d_optimizer
    #     self.g_optimizer = g_optimizer
    #     self.d_loss_fn = d_loss_fn
    #     self.g_loss_fn = g_loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        data_type = real_images.dtype
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), dtype=data_type)
        generated = self.generator(random_latent_vectors)
        real_labels, fakes_labels = tf.ones((batch_size, 1), dtype=data_type), -tf.ones((batch_size, 1), dtype=data_type)
        d_loss, g_loss, avg_d_loss, avg_g_loss = 0, 0, 0, 0
        lamb = 10 # tf.constant(10, dtype=data_type)
        for _ in range(self.discriminator_epochs):
            with tf.GradientTape() as tape:
                # d_loss = self.d_loss_fn(self.discriminator(real_images), self.discriminator(generated))
                d_loss = self.d_loss_fn(tf.concat((real_labels, fakes_labels), 0), tf.concat((self.discriminator(real_images), self.discriminator(generated)), 0))
                r = tf.random.uniform(shape=[1])
                x_hat = r*real_images + (1 - r)*generated
                val = lamb*((abs(tf.reduce_mean(x_hat) - tf.reduce_mean(self.discriminator(x_hat))))**2)
                d_loss += val
                avg_d_loss += d_loss
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        for _ in range(self.generator_epochs):
            with tf.GradientTape() as tape:
                predictions = self.discriminator(self.generator(random_latent_vectors))
                g_loss = self.g_loss_fn(real_labels, predictions) # g_loss = self.g_loss_fn(None, predictions)
                avg_g_loss += g_loss
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), dtype=data_type)
        # avg_d_loss, avg_g_loss = avg_d_loss / self.discriminator_epochs, avg_g_loss / self.generator_epochs
        fft_score = metric_fft_score(real_images, self.generator(random_latent_vectors))
        # tf.config.run_functions_eagerly(False)
        return {'d_loss': d_loss, 'g_loss': g_loss,
                'wasserstein_score': wasserstein_metric_fn(-2, self.discriminator(self.generator(random_latent_vectors))),
                'fft_score': fft_score}


if __name__ == '__main__':
    # tf.config.run_functions_eagerly(True)
    print(tf.executing_eagerly())
    data_type, batch_size = 'float32', 8
    # print(os.getcwd())
    # exit()
    time, benign_data = read_file_to_arrays('../signal_data/T04.txt')[0], [
        read_file_to_arrays(os.path.join('../signal_data', name))[1] for name in ['T04.txt',
                                                                                  'T04repeat.txt', 'T05.txt', 'T06.txt',
                                                                                  'T07.txt', 'T08.txt']]
    benign_data = np.array(benign_data[0:1]) # Training on one example to narrow down issue
    # time, benign_data = np.load('../signal_data/time_np.npy'), np.concatenate(
    #     [[np.load(os.path.join('../signal_data', name + '_np.npy'))] for name in ['T04',
    #                                                                               'T04repeat', 'T05', 'T06', 'T07',
    #                                                                               'T08']])

    # Repeat data to increase data size
    # benign_data = benign_data.repeat(1e3, axis=0)
    # print(benign_data.shape)

    # # Generating generic data for few shot learning
    # init_training = generate_generic_training_data(time, num_signals=100*5, frequencies=range(1, 11),
    #                                                amplitudes=range(1, 1001), h_offsets=[0.01 * n for n in range(0, 100)])
    # init_training = np.reshape(init_training, (-1, 5501, 5))
    # print(init_training.shape)
    # generic_dataset = data_to_dataset(init_training, dtype=data_type, batch_size=batch_size)

    benign_data_transposed = np.transpose(benign_data, (0, 2, 1))
    data_size, num_data_types = len(time), len(keys) - 1
    vector_size, latent_dimension = data_size, 256
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
    transformed = transformed.repeat(1e4, axis=0)
    print('Repeated shape: '.format(transformed.shape))
    # fs = data_size / (time[-1] - time[0])
    # print(fs)
    # temp_sig = signal.spectrogram(benign_data_transposed[0], fs)
    # print(temp_sig[2].shape)
    # plt.pcolormesh(temp_sig[2][0])
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
    dataset = data_to_dataset(transformed, dtype=data_type, batch_size=batch_size, shuffle=True)
    # dataset = data_to_dataset(benign_data, dtype=data_type, batch_size=batch_size, shuffle=True)
    # create discriminator and generator
    print(np.array(benign_data).shape)

    discriminator = make_AF_discriminator(num_data_types, data_size)
    generator = make_AF_generator(latent_dimension, num_data_types, data_size)
    # generator attempts to produce even numbers, discriminator will tell if true or not
    wgan = WGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dimension)
    wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
                 g_optimizer=keras.optimizers.Adam(learning_rate=0.0002)
                 )
    # exit()
    wgan.set_train_epochs(4, 1)
    # # prefit with generic data for few shot learning
    # wgan.fit(generic_dataset, epochs=128, batch_size=batch_size)
    # # Train model with real data
    wgan.fit(dataset, epochs=1, batch_size=batch_size)

    # Saving models
    # generator.save('af_generator')
    # discriminator.save('af_discriminator')
    rp = RecurrencePlot()
    prediction = generator.predict(tf.random.normal(shape=(1, latent_dimension)))[0].transpose(1, 0)
    print("Comparison shape: {}, Prediction shape: {}".format(transformed[0].transpose(1, 0)[0].shape, prediction[0].shape))
    for data_col in range(benign_data_transposed[0].shape[0]):
        temp_fig = plt.figure()
        ax = temp_fig.subplots(ncols=3, sharey=True)
        temp_orig = get_recurrence(benign_data_transposed[0, data_col], rp)
        temp_synth = get_recurrence(prediction[data_col], rp)
        temp_diff = temp_orig - temp_synth
        for idx, arr in enumerate((temp_orig, temp_synth, temp_diff)):
            ax[idx].imshow(arr, cmap='binary')
    temp_corr_fig = plt.figure()
    ax = temp_corr_fig.subplots(ncols=benign_data_transposed[0].shape[0], sharey=True)
    for data_col in range(benign_data_transposed[0].shape[0]):
        comp_sig = transformed[0].transpose(1, 0)[data_col]
        temp_orig = signal.correlate(comp_sig, comp_sig)
        temp_synth = signal.correlate(comp_sig, prediction[data_col])
        ax[data_col].plot(temp_orig)
        ax[data_col].plot(temp_synth)
    # ax[0].imshow(orig, cmap='binary')
    # # plt.show()
    # ax[1].imshow(synth, cmap='binary')
    # # plt.show()
    # ax[2].imshow(orig - synth, cmap='binary')
    # plt.savefig('./results/AF_RP_5_23_21.png')
    # plt.show()
    cross_corr_fig = plt.figure()
    ax = cross_corr_fig.subplots(nrows=3)
    ax[0].plot(signal.correlate(transformed.transpose(0, 2, 1)[0, 0], transformed.transpose(0, 2, 1)[0, 0]))
    ax[1].plot(signal.correlate(transformed.transpose(0, 2, 1)[0, 0], prediction[0]))
    ax[2].plot(signal.correlate(prediction[0], prediction[0]))
    prediction = generator.predict(tf.random.normal(shape=(1, latent_dimension)))[0]
    print(get_cross_correlate_score(benign_data_transposed, generator.predict(tf.random.normal(shape=(1, latent_dimension))).transpose(0, 2, 1)))
    print(get_fft_score(transformed[0:2].transpose(0, 2, 1), generator.predict(tf.random.normal(shape=(64, latent_dimension))).transpose(0, 2, 1)))  # time,
    # prediction = np.transpose(prediction, (0, 2, 1))
    # prediction = [scalers[idx].inverse_transform(benign_data_transposed[:][:][idx]) for idx in range(len(prediction[:][:]))]
    # prediction = np.transpose(prediction, (0, 2, 1))
    # plt.plot(time, transformed.transpose(0, 2, 1)[0][0])
    plot_data(time, np.transpose(np.array(prediction), (1, 0)), transformed.transpose(0, 2, 1)[0], show=True, save=False,
              save_path='./results/AF_5_23_21_')
