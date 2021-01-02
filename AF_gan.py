import numpy as np
from tensorflow.python.keras.backend import sign
from keras_gan import GAN
from keras_data import data_to_dataset, plot_data
import os
import tensorflow as tf

from custom_losses import GeneratorWassersteinLoss, DiscriminatorWassersteinLoss
from tensorflow import keras
from tensorflow.keras import layers


keys = ['time', 'accel', 'intaccel', 'sg1', 'sg2', 'sg3']

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
            # data['time'].append(float(split[0]))
            # data['accel'].append(float(split[1]))
            # data['intaccel'].append(float(split[2]))
            # data['sg1'].append(float(split[3]))
            # data['sg2'].append(float(split[4]))
            # data['sg3'].append(float(split[5]))
            data_line = tempFile.readline()
    return data

def signal_dict_to_list(data):
    """ Take signal dictionary and convert to list, time excluded"""
    # signal_data = [ data[val] for val in keys[1:]] # as 5 rows of 5501
    indices = [idx for idx in range(len(data['time']))]
    signal_data = list(map(lambda x: [data[key][x] for key in keys[1:]], indices))
    return signal_data # grab all but time list and make list


if __name__ == '__main__':
    data_type, batch_size = 'float32', 1
    # print(os.getcwd())
    # exit()
    signals = [ read_file(os.path.join('../signal_data', name)) for name in ['T04.txt',
        'T04repeat.txt', 'T05.txt', 'T06.txt', 'T07.txt', 'T08.txt']]
    # signals = list(map(read_file, signals))
    # for signal in signals:
    #     print(len(signal['time']))
    data_size, num_data_types = len(signals[0]['time']), len(keys) - 1
    vector_size, latent_dimension = data_size, 164
    benign_data = list(map(signal_dict_to_list, signals))
    # print('Length of data: {}, Number of data points: {}'.format(len(benign_data), len(benign_data[0])))
    # print(np.array(benign_data).shape)
    dataset = data_to_dataset(benign_data, dtype=data_type, batch_size=batch_size, shuffle=True)
    # create discriminator and generator
    discriminator = keras.Sequential(
    [
        # keras.Input(shape=(len(signals[0]['time']), len(keys) - 1, 1)),
        layers.Reshape((data_size, num_data_types, 1), input_shape=(data_size, num_data_types,)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", ),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
    )
    discriminator.summary()
    # print(np.array(benign_data).shape)
    # exit()

    # Create the generator
    # latent_dim = 128
    mini_data = int((data_size + 1) / 42)
    generator = keras.Sequential(
        [
            # keras.Input(shape=(latent_dimension,)),
            # We want to generate 128 coefficients to reshape into a 7x7x128 map
            layers.Dense(mini_data * num_data_types * latent_dimension, input_shape=(latent_dimension,)),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((num_data_types, mini_data, latent_dimension)),
            layers.Conv2DTranspose(64, (num_data_types, num_data_types), strides=(1, 6), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(128, (num_data_types, num_data_types), strides=(1, 7), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(1, (num_data_types, num_data_types), padding="same", activation="sigmoid"),
            layers.Cropping2D(((0, 0), (0, 1))),
            layers.Reshape((data_size, num_data_types))
        ],
        name="generator",
    )
    generator.add
    generator.summary()
    # print(np.array(benign_data).shape)
    # exit()
    # generator attempts to produce even numbers, discriminator will tell if true or not
    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dimension)
    gan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                # g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                # d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
                g_loss_fn=GeneratorWassersteinLoss(),
                d_loss_fn=DiscriminatorWassersteinLoss()
    )
    gan.fit(dataset, epochs=50)
    time = np.array(signals[0]['time'])
    plot_data(time, )