import os
from re import L

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.backend import sign
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization

from custom_losses import (DiscriminatorWassersteinLoss,
                           GeneratorWassersteinLoss)
from keras_data import data_to_dataset, plot_data
from keras_gan import WGAN

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
            time.append(float(split[0])) # append time to time column
            data.append([float(value) for value in split[1:]]) # append data as 5-tuple
            data_line = tempFile.readline()
    return time, data

def signal_dict_to_list(data):
    """ Take signal dictionary and convert to list, time excluded"""
    # signal_data = [ data[val] for val in keys[1:]] # as 5 rows of 5501
    indices = [idx for idx in range(len(data['time']))]
    signal_data = list(map(lambda x: [data[key][x] for key in keys[1:]], indices))
    return signal_data # grab all but time list and make list

# def convert_to_time_data

def plot_data(x_values, list_y_values, show=False, save=True, save_path=''):
    for idx in range(len(save_paths)):
        y_values = list_y_values[idx]
        print(y_values)
        print(y_values.shape)
        plt.plot(x_values, y_values)
        plt.title(label=save_path.split('/')[-1]+save_paths[idx])
        if save and save_path:
            plt.savefig(save_path+save_paths[idx])
        if show:
            plt.show()
        plt.close()

if __name__ == '__main__':
    data_type, batch_size = 'float32', 1
    # print(os.getcwd())
    # exit()
    time, benign_data = read_file_to_arrays('../signal_data/T04.txt')[0], [ read_file_to_arrays(os.path.join('../signal_data', name))[1] for name in ['T04.txt',
        'T04repeat.txt', 'T05.txt', 'T06.txt', 'T07.txt', 'T08.txt']]
    data_size, num_data_types = len(time), len(keys) -1
    vector_size, latent_dimension = data_size, 164
    dataset = data_to_dataset(benign_data, dtype=data_type, batch_size=batch_size, shuffle=True)
    # create discriminator and generator
    discriminator = keras.Sequential(
    [
        # keras.Input(shape=(len(signals[0]['time']), len(keys) - 1, 1)),
        layers.Reshape((vector_size, num_data_types, 1), input_shape=(vector_size, num_data_types,)),
        layers.Conv2D(16, (3, 1), strides=(3, 1)),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, (3, 1), strides=(3, 1)),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, (3, 1), strides=(3, 1)),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (3, 1), (1, 1)),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        # layers.GlobalMaxPooling2D(),
        layers.Dense(64),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(32),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1),
    ],
    name="discriminator",
    )
    discriminator.summary()
    print(np.array(benign_data).shape)
    # exit()

    # Create the generator
    # latent_dim = 128
    mini_data = 171
    generator = keras.Sequential(
        [
            # keras.Input(shape=(latent_dimension,)),
            # We want to generate 128 coefficients to reshape into a 7x7x128 map
            layers.Dense(mini_data * num_data_types, input_shape=(latent_dimension,)),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((mini_data, num_data_types, 1,)),
            layers.Conv2DTranspose(16, (3, 1), strides=(2, 1)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(16, (3, 1), strides=(2, 1)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(16, (3, 1), strides=(2, 1)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(16, (3, 1), strides=(2, 1)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(16, (3, 1), strides=(2, 1)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(1, (3, 1), activation='tanh'),
            layers.Reshape((data_size, num_data_types,))
        ],
        name="generator",
    )
    generator.summary()
    # generator attempts to produce even numbers, discriminator will tell if true or not
    wgan = WGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dimension)
    wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                g_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                # g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                # d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
                g_loss_fn=GeneratorWassersteinLoss(),
                d_loss_fn=DiscriminatorWassersteinLoss()
    )
    # exit()
    wgan.set_train_epochs(8, 1)
    wgan.fit(dataset, epochs=128, batch_size=batch_size)
    # generator.save('af_generator.h5')
    # discriminator.save('af_discriminator.h5')
    # time = np.array(signals[0]['time'])
    prediction = generator.predict(tf.random.normal(shape=(1, latent_dimension)))[0]
    plot_data(time, np.transpose(np.array(prediction), (1, 0)) , show=True, save=False, save_path='./keras_NN/results/AF_v1_')
