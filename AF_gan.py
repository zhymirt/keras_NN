import os

import numpy as np
from scipy import signal
import tensorflow as tf
from matplotlib import pyplot as plt
from pyts.image.recurrence import RecurrencePlot
from tensorflow import keras

from custom_losses import (DiscriminatorWassersteinLoss,
                           GeneratorWassersteinLoss)
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

def plot_data(x_values, list_y_values, show=False, save=True, save_path=''):
    for idx in range(len(save_paths)):
        y_values = list_y_values[idx]
        print(y_values)
        print(y_values.shape)
        plt.plot(x_values, y_values)
        plt.title(label=save_path.split('/')[-1] + save_paths[idx])
        if save and save_path:
            plt.savefig(save_path + save_paths[idx])
        if show:
            plt.show()
        plt.close()


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


if __name__ == '__main__':
    data_type, batch_size = 'float32', 6
    # print(os.getcwd())
    # exit()
    time, benign_data = read_file_to_arrays('../signal_data/T04.txt')[0], [
        read_file_to_arrays(os.path.join('../signal_data', name))[1] for name in ['T04.txt',
                                                                                  'T04repeat.txt', 'T05.txt', 'T06.txt',
                                                                                  'T07.txt', 'T08.txt']]
    benign_data = np.array(benign_data)
    # time, benign_data = np.load('../signal_data/time_np.npy'), np.concatenate(
    #     [[np.load(os.path.join('../signal_data', name + '_np.npy'))] for name in ['T04',
    #                                                                               'T04repeat', 'T05', 'T06', 'T07',
    #                                                                               'T08']])
    print(benign_data.shape)
    # # Repeat data to increase data size
    # benign_data = benign_data.repeat(1e3, axis=0)
    # print(benign_data.shape)

    # # Generating generic data for few shot learning
    # init_training = generate_generic_training_data(time, num_signals=100*5, frequencies=range(1, 11),
    #                                                amplitudes=range(1, 1001), h_offsets=[0.01 * n for n in range(0, 100)])
    # init_training = np.reshape(init_training, (-1, 5501, 5))
    # print(init_training.shape)
    # generic_dataset = data_to_dataset(init_training, dtype=data_type, batch_size=batch_size)

    benign_data_transposed = np.transpose(benign_data, (0, 2, 1))
    # print(np.load('../signal_data/T05_np.npy').shape)
    data_size, num_data_types = len(time), len(keys) - 1
    vector_size, latent_dimension = data_size, 256
    fs = data_size / (time[-1] - time[0])
    print(fs)
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
    dataset = data_to_dataset(benign_data, dtype=data_type, batch_size=batch_size, shuffle=True)
    # create discriminator and generator
    print(np.array(benign_data).shape)

    discriminator = make_AF_discriminator(num_data_types, data_size)
    generator = make_AF_generator(latent_dimension, num_data_types, data_size)
    # generator attempts to produce even numbers, discriminator will tell if true or not
    wgan = WGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dimension)
    wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                 g_optimizer=keras.optimizers.Adam(learning_rate=0.0006)
                 )
    # exit()
    wgan.set_train_epochs(4, 1)
    # # prefit with generic data for few shot learning
    # wgan.fit(generic_dataset, epochs=128, batch_size=batch_size)
    wgan.fit(dataset, epochs=1024, batch_size=batch_size)

    # Saving models
    # generator.save('af_generator.h5')
    # discriminator.save('af_discriminator.h5')
    # time = np.array(signals[0]['time'])
    rp = RecurrencePlot()
    orig = get_recurrence(benign_data_transposed[0][0], rp)
    synth = get_recurrence(generator.predict(tf.random.normal(shape=(1, latent_dimension)))[0].transpose(1, 0)[0], rp)
    plt.imshow(orig)
    plt.show()
    plt.imshow(synth)
    plt.show()
    plt.imshow(orig - synth)
    plt.show()
    prediction = generator.predict(tf.random.normal(shape=(1, latent_dimension)))[0]
    plot_data(time, np.transpose(np.array(prediction), (1, 0)), show=True, save=False,
              save_path='./keras_NN/results/AF_v1_')
