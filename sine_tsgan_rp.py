import os
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot as plt
from pyts.image.recurrence import RecurrencePlot
from tensorflow.keras import layers, mixed_precision
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from custom_losses import (DiscriminatorWassersteinLoss,
                           GeneratorWassersteinLoss)
from keras_data import data_to_dataset, plot_data
from keras_gan import GAN, WGAN, CWGAN
from keras_model_functions import get_recurrence, plot_recurrence
from model_architectures.sine_tsgan_rp_architecture import (
    make_sine_wgan_discriminator, make_sine_wgan_generator,
    make_spectrogram_discriminator, make_spectrogram_generator)
from sine_gan import (generate_conditional_image_summary,
                      generate_image_summary, generate_sine, plot_sine)

if __name__=='__main__':
    # Equivalent to the two lines above from tensorflow
    # mixed_precision.set_global_policy('mixed_float16')

    start_point, end_point, vector_size = 0, 2, 100
    latent_dimension, epochs, data_size, batch_size, data_type = 256, 32, int(1e4), 8, 'float32'
    save_desc = '__latent_dimension_{}_epochs_{}_data_size_{}_batch_size_{}_type_cnn_cnn'.format(latent_dimension, epochs, data_size, batch_size)
    early_stop = EarlyStopping(monitor='d_loss', mode='min', verbose=1, patience=3)
    checkpoint = ModelCheckpoint(filepath='./tmp/checkpoint', save_weights_only=True)
    tb = keras.callbacks.TensorBoard(log_dir='./log_dir', histogram_freq=1)
    callback_list = [checkpoint] # [early_stop, checkpoint, tb]
    benign_data, labels = [], []
    for _ in range(int(data_size)):
        frequency = 1 # randint(1, 3)
        benign_data.append(generate_sine(start_point, end_point, vector_size, frequency=frequency))
        labels.append([frequency])
    # benign_data = [generate_sine(start_point, end_point, vector_size, frequency=1) for _ in range(int(data_size))] # generate 100 points of sine wave
    time = np.linspace(start_point, end_point, vector_size)
    rp = RecurrencePlot(time_delay=((end_point-start_point)/vector_size), threshold=None, percentage=50)
    rec_plots = list(map(lambda x: get_recurrence(x, rp), benign_data))
    print("Recurrence plot Shape: {}\nBenign data Shape: {}".format(rec_plots[0].shape, benign_data[0].shape))
    rec_plot_dataset = data_to_dataset(rec_plots, dtype=data_type, batch_size=batch_size)
    dataset = data_to_dataset(benign_data, dtype=data_type, batch_size=batch_size, shuffle=True)
    # exit()
    discriminator_1 = make_spectrogram_discriminator(data_type=data_type)
    generator_1 = make_spectrogram_generator(latent_dimension, data_type=data_type)
    discriminator_2 = make_sine_wgan_discriminator(vector_size, data_type=data_type)
    generator_2 = make_sine_wgan_generator(latent_dimension, vector_size, data_type=data_type)
    # exit()
    spectrogram_wgan = WGAN(discriminator=discriminator_1, generator=generator_1, latent_dim=latent_dimension)
    spectrogram_wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.00006),
                g_optimizer=keras.optimizers.Adam(learning_rate=0.0003)
    )
    sine_wave_wgan = CWGAN(discriminator=discriminator_2, generator=generator_2, latent_dim=latent_dimension)
    sine_wave_wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                g_optimizer=keras.optimizers.Adam(learning_rate=0.0006)
    )
    time = np.linspace(start_point, end_point, num=vector_size)
    spectrogram_wgan.set_train_epochs(4, 1)
    # spectrogram_wgan.fit(rec_plot_dataset, epochs=2, batch_size=batch_size, callbacks=callback_list)
    # plt.imshow(rec_plots[0], cmap='binary')
    plt.show()
    # plt.imshow(generator_1.predict(tf.random.normal(shape=(1, latent_dimension)))[0], cmap='binary')
    # plt.savefig('4_20_21_synthetic_spectrogram')
    plt.show()
    # plt.pcolormesh(generator_1.predict(tf.random.normal(shape=(1, latent_dimension)))[0], cmap='binary')
    # plt.savefig('4_20_21_synthetic_spectrogram_1')
    plt.show()
    # exit()
    synthetic_spectrograms = np.array(generator_1.predict(tf.random.normal(shape=(data_size, latent_dimension))))
    # synthetic_spectrograms = np.array([generator_1.predict(tf.random.normal(shape=(1, latent_dimension)))[0] for _ in range(int(data_size))])
    benign_data = np.array(benign_data)
    # print("Length of benign: {} length of synthetic: {}".format(len(benign_data[0]), len(synthetic_spectrograms[0])))
    sine_wave_wgan.set_train_epochs(4, 1)
    sine_wave_wgan.fit((benign_data, synthetic_spectrograms), epochs=epochs, batch_size=batch_size, callbacks=callback_list)
    trend = generate_sine(start_point, end_point, vector_size)
    generate_conditional_image_summary(generator_2, generator_1.predict(tf.random.normal(shape=(1, latent_dimension))),
             latent_dimension, time, 3, True, trend, show=True, save=False, save_dir='./results', save_desc='4_9_21_tsgan_512_epochs')
