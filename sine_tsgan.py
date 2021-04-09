import os
from random import randint
from math import sqrt
from statistics import mean # , pstdev
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LeakyReLU, Dense, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPool1D
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.ops.gen_array_ops import Reshape
# from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
# from tensorflow.python.keras.layers.core import Dense
# from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
# from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPool1D
from keras_model_functions import plot_recurrence

from custom_losses import (DiscriminatorWassersteinLoss,
                           GeneratorWassersteinLoss)
from keras_data import data_to_dataset, plot_data
from keras_gan import GAN, WGAN, cWGAN
from sine_gan import generate_conditional_image_summary, generate_image_summary, generate_sine, plot_sine
from scipy import signal
import matplotlib.pyplot as plt

if __name__=='__main__':
    vector_size = 100
    start_point, end_point = 0, 2
    latent_dimension, epochs, data_size, batch_size, data_type = 256, 512, int(1e3), 8, 'float32'
    save_desc = '_{}{}{}{}{}{}{}{}{}{}'.format('latent_dimension_', latent_dimension, '_epochs_', epochs, '_data_size_', data_size, '_batch_size_', batch_size, '_type_', 'cnn_fc')
    early_stop = EarlyStopping(monitor='g_loss', mode='min', verbose=1, patience=3)
    checkpoint = ModelCheckpoint(filepath='./tmp/checkpoint', save_weights_only=True)
    callback_list = [checkpoint] # [early_stop, checkpoint]
    benign_data, labels = [], []
    for _ in range(int(data_size)):
        frequency = randint(1, 3)
        benign_data.append(generate_sine(start_point, end_point, vector_size, frequency=frequency))
        labels.append([frequency])
    # benign_data = [generate_sine(start_point, end_point, vector_size, frequency=1) for _ in range(int(data_size))] # generate 100 points of sine wave
    # spectrograms = list(map(lambda x: plt.specgram(x)[0], benign_data))
    spectrogram_scipy = list(map(lambda x: signal.spectrogram(x, fs=1)[2], benign_data))
    # plt.pcolormesh()
    # plt.close()
    # plt.plot(np.linspace(0, 2, 1000), benign_data[0]) 
    # plt.show()
    # plt.specgram(benign_data[0], Fs=1)
    # plt.show()
    # plt.close()

    # f, t, Sxx = signal.spectrogram(benign_data[0])
    # plt.pcolormesh(Sxx)
    # plt.savefig('sample')
    # plt.show()
    
    # plt.figure(figsize=(5, 4))
    # plt.imshow(Sxx, aspect='auto')
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    # img_data = plt.pcolormesh(t, f, Sxx, shading='gouraud').get_array()
    # plt.ylim((0, 3))
    # plt.show()
    # plt.pcolormesh(Sxx, shading='gouraud')
    # plt.ylim((0, 13))
    # plt.show()
    print("Spectrogram Shape: {}\nBenign data Shape: {}".format(spectrogram_scipy[0].shape, benign_data[0].shape))
    # spectrogram_dataset = tf.data.Dataset.from_tensor_slices(spectrogram_scipy)
    spectrogram_dataset = data_to_dataset(spectrogram_scipy, dtype=data_type, batch_size=batch_size)
    dataset = data_to_dataset(benign_data, dtype=data_type, batch_size=batch_size, shuffle=True)
    # exit()
    # print(spectrogram_scipy[0])
    image_shape, flattened_image_shape = (51, 1), (51,)
    discriminator_1 = keras.Sequential([
        layers.Reshape((1,)+image_shape, input_shape=image_shape),
        layers.Conv2D(16, (1, 5), strides=(1, 3)),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(16, (1, 3), strides=(1, 2)),
        layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(16, (1, 3), strides=(1, 2)),
        # layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (1, 3)),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dense(32),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1)
    ], name='discriminator_1')

    discriminator_2_spec_input = layers.Input(shape=image_shape)
    # discriminator_2_spec = layers.Embedding(10, 50)(discriminator_2_spec_input)
    discriminator_2_spec = layers.Reshape((1,)+image_shape)(discriminator_2_spec_input)
    discriminator_2_spec = layers.Conv2D(8, (1, 5), strides=(1, 3))(discriminator_2_spec)
    discriminator_2_spec = layers.LeakyReLU(alpha=0.2)(discriminator_2_spec)
    # discriminator_2_spec = layers.Conv2D(16, (1, 3), strides=(1, 2))(discriminator_2_spec)
    # discriminator_2_spec = layers.LeakyReLU(alpha=0.2)(discriminator_2_spec)
    discriminator_2_spec = layers.Conv2D(1, (1, 3), strides=(1, 2))(discriminator_2_spec)
    discriminator_2_spec = layers.LeakyReLU(alpha=0.2)(discriminator_2_spec)
    discriminator_2_spec = layers.Flatten()(discriminator_2_spec)
    discriminator_2_spec = layers.Dense(vector_size)(discriminator_2_spec)
    discriminator_2_spec = layers.Reshape((vector_size, 1))(discriminator_2_spec)

    discriminator_2_vector_input = layers.Input(shape=(vector_size,))
    discriminator_2_vector = layers.Reshape((vector_size, 1))(discriminator_2_vector_input)

    discriminator_2 = layers.Concatenate()([discriminator_2_vector, discriminator_2_spec])
    discriminator_2 = layers.Conv1D(16, 5, strides=3)(discriminator_2)
    discriminator_2 = layers.LeakyReLU(alpha=0.2)(discriminator_2)
    discriminator_2 = layers.Conv1D(16, 3, strides=2)(discriminator_2)
    discriminator_2 = layers.LeakyReLU(alpha=0.2)(discriminator_2)
    discriminator_2 = layers.Conv1D(1, 3)(discriminator_2)
    discriminator_2 = layers.LeakyReLU(alpha=0.2)(discriminator_2)
    discriminator_2 = layers.Flatten()(discriminator_2)
    discriminator_2 = layers.Dense(32)(discriminator_2)
    discriminator_2 = layers.Dense(1)(discriminator_2)
    discriminator_2 = keras.Model(inputs=(discriminator_2_vector_input, discriminator_2_spec_input), outputs=discriminator_2, name="discriminator_2")
    # discriminator_2 = keras.Sequential([
    #     layers.Reshape((vector_size, 1,), input_shape=(vector_size,)),
    #     layers.Conv1D(64, 5, strides=3),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Conv1D(64, 3, strides=2),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Conv1D(1, 3),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Flatten(),
    #     layers.Dense(32),
    #     layers.Dense(1)
    # ], name='discriminator_2')
    generator_1 = keras.Sequential([
        layers.Dense(12, input_shape=(latent_dimension,)),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((1, 12, 1,)),
        layers.Conv2DTranspose(32, (1, 3), strides=(1, 2)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(1, (1, 3), strides=(1, 2)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(1, 3),
        # layers.LeakyReLU(alpha=0.2),
        layers.Reshape(image_shape),
    ], name='generator_1')

    generator_2_spec_input = layers.Input(shape=image_shape)
    # generator_2_spec = layers.Embedding(10, 20)(generator_2_spec_input)
    # generator_2_spec = layers.Flatten()(generator_2_spec_input) # (generator_2_spec)
    # generator_2_spec = layers.Dense(flattened_image_shape[0])(generator_2_spec)
    generator_2_spec = layers.Reshape((1,)+image_shape)(generator_2_spec_input)

    generator_2_vec_input = layers.Input((latent_dimension,))
    generator_2_vec = layers.Dense(flattened_image_shape[0])(generator_2_vec_input)
    generator_2_vec = layers.Reshape((1,)+image_shape)(generator_2_vec)

    generator_2 = layers.Concatenate()([generator_2_vec, generator_2_spec])
    generator_2 = layers.Conv2D(16, (1, 3), strides=(1, 2))(generator_2)
    generator_2 = layers.Conv2D(16, (1, 3), strides=(1, 2))(generator_2)
    generator_2 = layers.Conv2D(1, (1, 3))(generator_2)
    generator_2 = layers.Flatten()(generator_2)
    generator_2 = layers.Dense(64)(generator_2)
    generator_2 = layers.LeakyReLU(alpha=0.2)(generator_2)
    generator_2 = layers.Reshape((64, 1))(generator_2)
    generator_2 = layers.Conv1DTranspose(16, 3, strides=3, dtype=data_type)(generator_2)
    generator_2 = layers.BatchNormalization()(generator_2)
    generator_2 = layers.LeakyReLU(alpha=0.2, dtype=data_type)(generator_2)
    generator_2 = layers.Conv1DTranspose(16, 3, strides=3, dtype=data_type)(generator_2)
    generator_2 = layers.BatchNormalization()(generator_2)
    generator_2 = layers.LeakyReLU(alpha=0.2, dtype=data_type)(generator_2)
    generator_2 = layers.Conv1D(1, 3, dtype=data_type)(generator_2)
    generator_2 = layers.Flatten()(generator_2)
    generator_2 = layers.Dense(32, activation=tf.cos, dtype=data_type)(generator_2)
    generator_2 = layers.Dense(vector_size, activation='tanh', dtype=data_type)(generator_2)
    generator_2 = keras.Model(inputs=(generator_2_vec_input, generator_2_spec_input), outputs=generator_2, name="generator_2")
    # generator_2 = keras.Sequential([
    #     layers.Reshape((1,)+image_shape, input_shape=image_shape),
    #     layers.Conv1D(16, 3, strides=2),
    #     layers.Conv1D(16, 3, strides=2),
    #     layers.Conv1D(1, 3),
    #     layers.Flatten(),
    #     layers.Dense(64),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Reshape((64, 1)),
    #     layers.Conv1DTranspose(16, 3, strides=3, dtype=data_type),
    #     layers.BatchNormalization(),
    #     layers.LeakyReLU(alpha=0.2, dtype=data_type),
    #     layers.Conv1DTranspose(16, 3, strides=3, dtype=data_type),
    #     layers.BatchNormalization(),
    #     layers.LeakyReLU(alpha=0.2, dtype=data_type),
    #     layers.Conv1D(1, 3, dtype=data_type),
    #     layers.Flatten(),
    #     layers.Dense(32, activation=tf.cos, dtype=data_type),
    #     layers.Dense(vector_size, activation='tanh', dtype=data_type)
    # ], name='generator_2')
    print(discriminator_1.input_shape)
    print(discriminator_2.input_shape)
    print(generator_1.input_shape)
    print(generator_2.input_shape)
    print(discriminator_1.summary())
    print(discriminator_2.summary())
    print(generator_1.summary())
    print(generator_2.summary())
    print(discriminator_1.output_shape)
    print(discriminator_2.output_shape)
    print(generator_1.output_shape)
    print(generator_2.output_shape)
    spectrogram_wgan = WGAN(discriminator=discriminator_1, generator=generator_1, latent_dim=latent_dimension)
    spectrogram_wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                g_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                # g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                # d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
                g_loss_fn=GeneratorWassersteinLoss(),
                d_loss_fn=DiscriminatorWassersteinLoss()
    )
    sine_wave_wgan = cWGAN(discriminator=discriminator_2, generator=generator_2, latent_dim=latent_dimension)
    sine_wave_wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                g_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                # g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                # d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
                g_loss_fn=GeneratorWassersteinLoss(),
                d_loss_fn=DiscriminatorWassersteinLoss()
    )
    time = np.linspace(start_point, end_point, num=vector_size)
    spectrogram_wgan.set_train_epochs(5, 1)
    spectrogram_wgan.fit(spectrogram_dataset, epochs=epochs, batch_size=batch_size, callbacks=callback_list)
    plt.pcolormesh(spectrogram_scipy[0])
    plt.show()
    plt.pcolormesh(generator_1.predict(tf.random.normal(shape=(1, latent_dimension)))[0])
    plt.savefig('synthetic_spectrogram_512_epocs')
    plt.show()
    synthetic_spectrograms = np.array([generator_1.predict(tf.random.normal(shape=(1, latent_dimension)))[0] for _ in range(int(data_size))])
    benign_data = np.array(benign_data)
    # print("Length of benign: {} length of synthetic: {}".format(len(benign_data[0]), len(synthetic_spectrograms[0])))
    sine_wave_wgan.set_train_epochs(4, 1)
    sine_wave_wgan.fit((benign_data, synthetic_spectrograms), epochs=epochs, batch_size=batch_size, callbacks=callback_list)
    trend = generate_sine(start_point, end_point, vector_size)
    generate_conditional_image_summary(generator_2, generator_1.predict(tf.random.normal(shape=(1, latent_dimension))),
             latent_dimension, time, 3, True, trend, show=True, save=True, save_dir='./results', save_desc='4_9_21_tsgan_512_epochs')
