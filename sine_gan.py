from random import randint
from math import sqrt
from statistics import mean # , pstdev
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPool1D

from custom_losses import (DiscriminatorWassersteinLoss,
                           GeneratorWassersteinLoss)
from keras_data import data_to_dataset, plot_data
from keras_gan import GAN, WGAN


start_point, end_point, vector_size = 0, 2, 1000

def generate_sine(start, end, points, amplitude=1, frequency=1):
    time = np.linspace(start_point, end_point, vector_size)
    signal = amplitude*np.sin(2*np.pi*frequency*time)
    return signal

def plot_sine(data, show=False, save=False, save_path=''):
    plot_data(np.linspace(start_point, end_point, num=vector_size), data, show=show, save=save, save_path=save_path)

# def mean(vector):
#     return sum(vector)/len(vector) if vector else 0

def std_dev(vector):
    if vector is None or len(vector) == 0:
        return 0
    size = len(vector)
    avg = sum(vector)/size
    return sqrt(sum(list(map(lambda x: (x - avg)**2, vector)))/size)

def standardize(vector):
    avg = mean(vector)
    deviation = std_dev(vector)
    return list(map(lambda x: (x - avg / deviation) if deviation != 0 else 0, vector))


if __name__ == '__main__':
    linux = False
    if linux:
        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import InteractiveSession

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
    latent_dimension, epochs, data_size, batch_size, data_type = 256, 32, int(1e4), 32, 'float32'
    save_desc = '_{}{}{}{}{}{}{}{}{}{}'.format('latent_dimension_', latent_dimension, '_epochs_', epochs, '_data_size_', data_size, '_batch_size_', batch_size, '_type_', 'cnn_fc')
    early_stop = EarlyStopping(monitor='g_loss', mode='min', verbose=1, patience=3)
    checkpoint = ModelCheckpoint(filepath='./tmp/checkpoint', save_weights_only=True)
    callback_list = [checkpoint] # [early_stop, checkpoint]
    benign_data = [generate_sine(start_point, end_point, vector_size, frequency=randint(1, 3)) for _ in range(int(data_size))] # generate 100 points of sine wave
    for idx in range(2):
        plot_sine(benign_data[idx], show=True)
    dataset = data_to_dataset(benign_data, dtype=data_type, batch_size=batch_size, shuffle=True)
    # create discriminator and generator
    discrim = keras.Sequential(
    [
        layers.Reshape((vector_size, 1,), input_shape=(vector_size,)),
        layers.Conv1D(256, (5), strides=(2), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(256, (3), strides=(2), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(256, (3), strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalAveragePooling1D(),
        # layers.GlobalMaxPooling1D(),
        layers.Flatten(),
        # layers.Dense(3000),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Dense(2000),
        # layers.LeakyReLU(alpha=0.2),
        layers.Dense(1) #, activation='sigmoid'),
    ],
    name="discriminator",
    )
    # discrim = keras.Sequential([
    #     layers.Dense(vector_size*4, input_shape=(vector_size,)),
    #     layers.Dense(1024),
    #     layers.BatchNormalization(),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Dense(1024),
    #     layers.BatchNormalization(),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Dense(1024),
    #     layers.BatchNormalization(),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Dense(1024),
    #     layers.BatchNormalization(),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Dense(1)
    # ],
    # name="discriminator",
    # )
    discrim.summary()
    # exit()
    generator = keras.Sequential(
        [
            # layers.Dense(vector_size * latent_dimension, input_shape=(latent_dimension,)),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Dense(latent_dimension),
            # layers.LeakyReLU(alpha=0.2),
            layers.Reshape((latent_dimension, 1), input_shape=(latent_dimension,)),
            layers.Conv1DTranspose(256, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(256, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1D(1, (3), padding='same', activation='tanh'),
            layers.BatchNormalization(),
            # layers.Reshape((vector_size,)),
            # layers.Reshape((4 * vector_size,)),
            layers.Flatten(),
            # layers.Dense(4000, activation='sigmoid'),

            # layers.Dense(2000, activation='relu'),
            # layers.Dense(vector_size, activation='relu')
            layers.Dense(vector_size, activation=tf.cos),
            layers.BatchNormalization(),
            layers.Dense(vector_size, activation='tanh')
        ],
        name="generator",
    )
    generator = keras.Sequential(
        [
            layers.Dense(vector_size * latent_dimension, input_shape=(latent_dimension,)),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.Dense(512, activation=tf.cos),
            layers.Dense(vector_size, activation='tanh')
        ],
        name="generator",
    )
    generator.summary()
    # d_optimizer = keras.optimizers.Adam(learning_rate=0.0003)
    # exit()
    # generator attempts to produce even numbers, discriminator will tell if true or not
    wgan = WGAN(discriminator=discrim, generator=generator, latent_dim=latent_dimension)
    wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                # g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                # d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
                g_loss_fn=GeneratorWassersteinLoss(),
                d_loss_fn=DiscriminatorWassersteinLoss()
    )
    # print(len(benign_data))
    # discrim.compile(d_optimizer, keras.losses.BinaryCrossentropy(from_logits=True))
    # discrim.fit(np.array(benign_data), np.array([[int(1)] for _ in range(data_size)]), epochs=3)
    wgan.set_train_epochs(5, 1)
    wgan.fit(dataset, epochs=epochs, callbacks=callback_list)
    # generator.save('sine_generator.h5')
    # discrim.save('sine_discriminator.h5')
    plot_sine(generator.predict(tf.zeros(shape=(1, latent_dimension)))[0], show=True, save=True, save_path='./results/sine_zeros' + save_desc)
    plot_sine(standardize(generator.predict(tf.zeros(shape=(1, latent_dimension)))[0]), show=True, save=False)
    for idx in range(3):
        plot_sine(generator.predict(tf.random.normal(shape=(1, latent_dimension)))[0], show=True, save=True, save_path='./results/sine_norm' + save_desc + '_' + str(idx))
    plot_sine(generator.predict(tf.ones(shape=(1, latent_dimension)))[0], show=True, save=True, save_path='./results/sine_ones' + save_desc)
