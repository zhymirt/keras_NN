import os
from random import randint
from math import sqrt
from statistics import mean  # , pstdev
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LeakyReLU, Dense, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPool1D
from tensorflow.keras.layers import Conv1D, Flatten, Reshape
from tensorflow.keras import mixed_precision
from keras_model_functions import plot_recurrence

from custom_losses import (DiscriminatorWassersteinLoss,
                           GeneratorWassersteinLoss)
from keras_data import data_to_dataset, plot_data
from custom_classes import GAN, WGAN, CWGAN
from model_architectures.sine_tsgan_architecture import make_sine_tsgan_discriminator_1, make_sine_tsgan_generator_1, \
    make_sine_tsgan_discriminator_2, make_sine_tsgan_generator_2
from sine_gan import generate_conditional_image_summary, generate_image_summary, generate_sine, plot_sine
from scipy import signal
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Equivalent to the two lines above from tensorflow
    # mixed_precision.set_global_policy('mixed_float16')

    start_point, end_point, vector_size = 0, 2, 100
    latent_dimension, epochs, data_size, batch_size, data_type = 256, 128, int(1e3), 8, 'float32'
    save_desc = '_latent_dimension_{}_epochs_{}_data_size_{}_batch_size_{}_type_cnn_cnn'.format(latent_dimension,
                                                                                                epochs, data_size,
                                                                                                batch_size)
    early_stop = EarlyStopping(monitor='d_loss', mode='min', verbose=1, patience=3)
    checkpoint = ModelCheckpoint(filepath='./tmp/checkpoint', save_weights_only=True)
    callback_list = [checkpoint]  # [early_stop, checkpoint]
    benign_data, labels = [], []
    for idx in range(int(data_size)):
        frequency = randint(1, 2)  # (idx % 2) + 1
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
    discriminator_1 = make_sine_tsgan_discriminator_1()
    generator_1 = make_sine_tsgan_generator_1(latent_dimension)
    discriminator_2 = make_sine_tsgan_discriminator_2(vector_size)
    generator_2 = make_sine_tsgan_generator_2(latent_dimension, vector_size)

    spectrogram_wgan = WGAN(discriminator=discriminator_1, generator=generator_1, latent_dim=latent_dimension)
    spectrogram_wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.000006),
                             g_optimizer=keras.optimizers.Adam(learning_rate=0.00006),
                             # g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                             # d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
                             g_loss_fn=GeneratorWassersteinLoss(),
                             d_loss_fn=DiscriminatorWassersteinLoss()
                             )
    sine_wave_wgan = CWGAN(discriminator=discriminator_2, generator=generator_2, latent_dim=latent_dimension)
    sine_wave_wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.000006),
                           g_optimizer=keras.optimizers.Adam(learning_rate=0.00006),
                           # g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                           # d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
                           g_loss_fn=GeneratorWassersteinLoss(),
                           d_loss_fn=DiscriminatorWassersteinLoss()
                           )
    time = np.linspace(start_point, end_point, num=vector_size)
    spectrogram_wgan.set_train_epochs(4, 1)
    spectrogram_wgan.fit(spectrogram_dataset, epochs=epochs, batch_size=batch_size, callbacks=callback_list)
    plt.pcolormesh(spectrogram_scipy[0])
    plt.show()
    plt.pcolormesh(generator_1.predict(tf.random.normal(shape=(1, latent_dimension)))[0])
    plt.savefig('./results/4_23_21_synthetic_spectrogram_2_freq_1')
    plt.show()
    plt.pcolormesh(generator_1.predict(tf.random.normal(shape=(1, latent_dimension)))[0])
    plt.savefig('./results/4_23_21_synthetic_spectrogram_2_freq_2')
    plt.show()
    # exit()
    discriminator_1.save('./models/sine_tsgan_discriminator_1')
    generator_1.save('./models/sine_tsgan_generator_1')
    synthetic_spectrograms = np.array(
        [generator_1.predict(tf.random.normal(shape=(1, latent_dimension)))[0] for _ in range(int(data_size))])
    benign_data = np.array(benign_data)
    # print("Length of benign: {} length of synthetic: {}".format(len(benign_data[0]), len(synthetic_spectrograms[0])))
    sine_wave_wgan.set_train_epochs(4, 1)
    sine_wave_wgan.fit((benign_data, synthetic_spectrograms), epochs=epochs, batch_size=batch_size,
                       callbacks=callback_list)
    discriminator_2.save('./models/sine_tsgan_discriminator_2')
    generator_2.save('./models/sine_tsgan_generator_2')
    trend = generate_sine(start_point, end_point, vector_size)
    generate_conditional_image_summary(generator_2, generator_1.predict(tf.random.normal(shape=(1, latent_dimension))),
                                       latent_dimension, time, 3, True, trend, show=True, save=True,
                                       save_dir='./results', save_desc='4_23_21_tsgan_512_epochs')
