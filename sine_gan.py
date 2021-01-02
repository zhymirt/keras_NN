from random import randint

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.pooling import GlobalMaxPool1D

from custom_losses import (DiscriminatorWassersteinLoss,
                           GeneratorWassersteinLoss)
from keras_data import data_to_dataset, plot_data
from keras_gan import GAN


start_point, end_point, vector_size = 0, 2, 1000

def generate_sine(start, end, points, amplitude=1, frequency=1):
    time = np.linspace(start_point, end_point, vector_size)
    signal = amplitude*np.sin(2*np.pi*frequency*time)
    return signal

def plot_sine(data, show=False, save=False, save_path=''):
    plot_data(np.linspace(start_point, end_point, num=vector_size), data, show=show, save=save, save_path=save_path)

if __name__ == '__main__':
    latent_dimension, variations, data_size, batch_size, data_type = 128, 100, 1e4, 64, 'float32'
    benign_data = [generate_sine(start_point, end_point, vector_size, frequency=randint(1, 3)) for _ in range(int(data_size))] # generate 100 points of sine wave
    for idx in range(4):
        plot_sine(benign_data[idx], show=False)
    dataset = data_to_dataset(benign_data, dtype=data_type, batch_size=batch_size, shuffle=True)
    # create discriminator and generator
    discrim = keras.Sequential(
    [
        layers.Reshape((vector_size, 1,), input_shape=(vector_size,)),
        layers.Conv1D(64, (3), strides=(3), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(128, (3), strides=(3), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling1D(),
        layers.Reshape((128,)),
        layers.Dense(1, activation='sigmoid'),
    ],
    name="discriminator",
    )
    # print(discrim.input_shape, ' ', discrim.output_shape)
    # exit()
    generator = keras.Sequential(
        [
            layers.Dense(int(vector_size * 0.25) * latent_dimension, input_shape=(latent_dimension,)),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((-1, latent_dimension)),
            layers.Conv1DTranspose(128, 7, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(256, 7, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1D(1, (vector_size), padding='same', activation='sigmoid'),
            layers.Reshape((vector_size,))
            # layers.Reshape((4 * vector_size,)),
        #     layers.Dense(300),
        #     layers.Dense(200, activation=tf.math.cos),
        #     layers.Dense(vector_size, activation='tanh')
        ],
        name="generator",
    )
    print(generator.input_shape, ' ', generator.output_shape)
    generator.summary()
    # exit()
    # generator attempts to produce even numbers, discriminator will tell if true or not
    gan = GAN(discriminator=discrim, generator=generator, latent_dim=latent_dimension)
    gan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                # g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                # d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
                g_loss_fn=GeneratorWassersteinLoss(),
                d_loss_fn=DiscriminatorWassersteinLoss()
    )
    gan.fit(dataset, epochs=50)
    generator.save('sine_generator.h5')
    discrim.save('sine_discriminator.h5')
    plot_sine(generator.predict(tf.zeros(shape=(1, latent_dimension)))[0], show=True, save=True, save_path='./results/sine_zeros')
    for idx in range(3):
        plot_sine(generator.predict(tf.random.normal(shape=(1, latent_dimension)))[0], show=True, save=True, save_path='./results/sine_norm'+str(idx))
    plot_sine(generator.predict(tf.ones(shape=(1, latent_dimension)))[0], show=True, save=True, save_path='./results/sine_ones')
