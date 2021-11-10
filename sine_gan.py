import os
from math import sqrt
from random import randint
from statistics import mean  # , pstdev

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import sklearn.preprocessing as preprocessing
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import (BatchNormalization, Dense,
                                     GlobalAveragePooling1D, GlobalMaxPool1D,
                                     LeakyReLU)
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from custom_losses import (DiscriminatorWassersteinLoss,
                           GeneratorWassersteinLoss)
from keras_data import data_to_dataset, plot_data
from keras_gan import GAN, WGAN, CWGAN
from keras_model_functions import plot_recurrence
from model_architectures.sine_gan_architecture import (make_sine_gan_cnn_discriminator, make_sine_gan_fcc_discriminator,
                                                       make_sine_gan_cnn_generator, make_sine_gan_fcc_generator,
                                                       make_conditional_sine_gan_cnn_discriminator,
                                                       make_conditional_sine_gan_cnn_generator)


def generate_sine(start, end, points, amplitude=1, frequency=1):
    time = np.linspace(start, end, points)
    signal = amplitude * np.sin(2 * np.pi * frequency * time)
    return signal


def plot_sine(data, time, show=False, save=False, save_path='', plot_trend=False, trend_signal=None):
    plot_data(time,
              data, trend_signal if plot_trend and trend_signal is not None and len(trend_signal) > 0 else None,
              show=show, save=save, save_path=save_path)


def generate_image_summary(generator, latent_dim_size, time, num_rand_images=1, plot_trend=False, trend_signal=None,
                           show=False, save=False, save_dir='./', save_desc=''):
    get_data = lambda gen, input_data: gen.predict(input_data(shape=(1, latent_dim_size)))[0]
    plot_sine(get_data(generator, tf.zeros), time, show=show, save=save,
              save_path=os.path.join(save_dir, 'sine_zeros' + save_desc),
              plot_trend=plot_trend, trend_signal=trend_signal)
    for idx in range(num_rand_images):
        plot_sine(get_data(generator, tf.random.normal), time, show=True, save=save,
                  save_path=os.path.join(save_dir, 'sine_norm{}_{}'.format(save_desc, str(idx))),
                  plot_trend=plot_trend, trend_signal=trend_signal)
    plot_sine(get_data(generator, tf.ones), time, show=show, save=save,
              save_path=os.path.join(save_dir, 'sine_ones' + save_desc),
              plot_trend=plot_trend, trend_signal=trend_signal)


def generate_conditional_image_summary(generator, conditional_data, latent_dim_size, time, num_rand_images=1,
                                       plot_trend=False, trend_signal=None, show=False, save=False, save_dir='./',
                                       save_desc=''):
    get_data = lambda gen, input_data, cond: gen.predict((input_data(shape=(1, latent_dim_size)), cond))[0]
    plot_sine(get_data(generator, tf.zeros, conditional_data), time, show=show, save=save,
              save_path=os.path.join(save_dir, 'sine_zeros' + save_desc),
              plot_trend=plot_trend, trend_signal=trend_signal)
    for idx in range(num_rand_images):
        plot_sine(get_data(generator, tf.random.normal, conditional_data), time, show=True, save=save,
                  save_path=os.path.join(save_dir, 'sine_norm{}_{}'.format(save_desc, str(idx))),
                  plot_trend=plot_trend, trend_signal=trend_signal)
    plot_sine(get_data(generator, tf.ones, conditional_data), time, show=show, save=save,
              save_path=os.path.join(save_dir, 'sine_ones' + save_desc),
              plot_trend=plot_trend, trend_signal=trend_signal)


# def mean(vector):
#     return sum(vector)/len(vector) if vector else 0

def std_dev(vector):
    if vector is None or len(vector) == 0:
        return 0
    size = len(vector)
    avg = sum(vector) / size
    return sqrt(sum(list(map(lambda x: (x - avg) ** 2, vector))) / size)


def standardize(vector):
    avg = mean(vector)
    deviation = std_dev(vector)
    return list(map(lambda x: (x - avg / deviation) if deviation != 0 else 0, vector))


if __name__ == '__main__':
    # linux = False
    # if linux:
    #     from tensorflow.compat.v1 import ConfigProto
    #     from tensorflow.compat.v1 import InteractiveSession

    #     config = ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     session = InteractiveSession(config=config)
    start_point, end_point, vector_size = 0, 2, 6000
    conditional, discriminator_mode, generator_mode, verbosity = False, 'cnn', 'cnn', 2
    latent_dimension, epochs, data_size, batch_size, data_type = 256, 16, int(1e4), 16, 'float32'
    save_desc = '__latent_dimension_{}_epochs_{}_data_size_{}_batch_size_{}_type_cnn_cnn'.format(latent_dimension,
                                                                                                 epochs, data_size,
                                                                                                 batch_size)
    early_stop = EarlyStopping(monitor='g_loss', mode='min', min_delta=1e-8, verbose=1, patience=3)
    checkpoint = ModelCheckpoint(filepath='./tmp/checkpoint', save_weights_only=True)
    tb = keras.callbacks.TensorBoard(log_dir='./log_dir', histogram_freq=1)
    callback_list = [checkpoint, tb]  # [early_stop, checkpoint]
    benign_data = [generate_sine(start_point, end_point, vector_size, amplitude=randint(1e4, 1e4), frequency=randint(1, 2)) for _ in
                   range(int(data_size))]  # generate 100 points of sine wave
    # for idx in range(2):
    #     plot_sine(benign_data[idx], show=True)
    # scaler = preprocessing.MinMaxScaler().fit(benign_data)
    # transformed = scaler.transform(benign_data)
    # dataset = data_to_dataset(transformed, dtype=data_type, batch_size=batch_size, shuffle=True)
    dataset = data_to_dataset(benign_data, dtype=data_type, batch_size=batch_size, shuffle=True)
    # create discriminator and generator
    if conditional:
        save_desc = '__review_conditional_latent_dimension_{}_epochs_{}_data_size_{}_batch_size_{}_type_cnn_cnn'.format(
            latent_dimension, epochs, data_size, batch_size)
        benign_data, labels = [], []
        for _ in range(int(data_size)):
            frequency = randint(1, 3)
            benign_data.append(generate_sine(start_point, end_point, vector_size, amplitude=1, frequency=frequency))
            labels.append([frequency])
        # for idx in range(2):
        #     plot_sine(benign_data[idx], show=True)
        # dataset = data_to_dataset(benign_data, dtype=data_type, batch_size=batch_size, shuffle=True)
        benign_data, labels = np.array(benign_data), np.array(labels)
        num_frequencies = 4
        discriminator = make_conditional_sine_gan_cnn_discriminator(vector_size, num_frequencies)
        generator = make_conditional_sine_gan_cnn_generator(latent_dimension, vector_size, num_frequencies)
        cwgan = CWGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dimension)
        cwgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                      g_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                      # g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                      # d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
                      # g_loss_fn=GeneratorWassersteinLoss(),
                      # d_loss_fn=DiscriminatorWassersteinLoss()
                      )
        cwgan.set_train_epochs(5, 1)
        generator.predict((tf.zeros(shape=(1, latent_dimension)), tf.constant([1])))
        cwgan.fit((benign_data, labels), epochs=epochs, batch_size=batch_size, callbacks=callback_list)
        # cwgan.fit(x=benign_data, y=labels, epochs=epochs, batch_size=batch_size, callbacks=callback_list)
        generator.save('./models/conditional_sine_generator')
        discriminator.save('./models/conditional_sine_discriminator')
        time = np.linspace(start_point, end_point, num=vector_size)
        from pyts.image import RecurrencePlot

        rp, trend = RecurrencePlot(threshold='point', percentage=20), generate_sine(start_point, end_point, vector_size,
                                                                                    amplitude=1, frequency=1)
        plot_recurrence(generator.predict((tf.zeros(shape=(1, latent_dimension)), tf.constant([1])))[0], rp, show=True,
                        save=False)
        plot_sine(generator.predict([tf.zeros(shape=(1, latent_dimension)), tf.constant([1])])[0], time, show=True,
                  save=False, save_path='./results/sine_zeros' + save_desc)
        for idx in range(3):
            plot_sine(generator.predict((tf.random.normal(shape=(1, latent_dimension)), tf.constant([idx + 1])))[0],
                      time, show=True, save=False, save_path='./results/sine_norm' + save_desc + '_' + str(idx))
        plot_sine(generator.predict((tf.ones(shape=(1, latent_dimension)), tf.constant([1])))[0], time, show=True,
                  save=False, save_path='./results/sine_ones' + save_desc)
        generate_conditional_image_summary(generator=generator, conditional_data=tf.constant([1]), time=time,
                                           latent_dim_size=latent_dimension, num_rand_images=3, show=True, save=False,
                                           save_dir='./results', save_desc=save_desc, plot_trend=True,
                                           trend_signal=trend)
        exit()
    else:
        if discriminator_mode == 'cnn':
            discrim = make_sine_gan_cnn_discriminator(vector_size)
        else:
            discrim = make_sine_gan_fcc_discriminator(vector_size)
        if generator_mode == 'cnn':
            generator = make_sine_gan_cnn_generator(latent_dimension)
        else:
            generator = make_sine_gan_fcc_generator(latent_dimension, vector_size)
        # generator attempts to produce sine wave, discriminator will give score of authenticity
        wgan = WGAN(discriminator=discrim, generator=generator, latent_dim=latent_dimension)
        wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
                     g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
                     # g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                     # d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
                     g_loss_fn=GeneratorWassersteinLoss(),
                     d_loss_fn=DiscriminatorWassersteinLoss()
                     )
        wgan.set_train_epochs(4, 1)
        wgan.fit(dataset, epochs=epochs, batch_size=batch_size, callbacks=callback_list)
        # generator.save('./models/sine_generator_cnn_freq_1-3')
        # discrim.save('./models/sine_discriminator_cnn_freq_1-3')
        time = np.linspace(start_point, end_point, num=vector_size)
        from pyts.image import RecurrencePlot

        rp, trend = RecurrencePlot(), generate_sine(start_point, end_point, vector_size,
                                                    amplitude=int(1), frequency=1)
        plot_recurrence(generator.predict(tf.zeros(shape=(1, latent_dimension)))[0], rp, show=True, save=False,
                        save_name='5_7_21_sine_gan_freq_1-3_recurrence')
        generate_image_summary(generator=generator, time=time, latent_dim_size=latent_dimension, num_rand_images=3,
                               show=True, save=False,
                               save_dir='./results', save_desc='5_7_21_freq_1-3_' + save_desc, plot_trend=True, trend_signal=trend)
        # plot_sine(scaler.inverse_transform(generator.predict(tf.zeros(shape=(1, latent_dimension))))[0], time, show=True, save=False, save_path='./results/sine_zeros' + save_desc)
        # # plot_sine(standardize(generator.predict(tf.zeros(shape=(1, latent_dimension))))[0], time, show=True, save=False)
        # for idx in range(3):
        #     plot_sine(scaler.inverse_transform(generator.predict(tf.random.normal(shape=(1, latent_dimension))))[0], time, show=True, save=False, save_path='./results/sine_norm' + save_desc + '_' + str(idx))
        # plot_sine(scaler.inverse_transform(generator.predict(tf.ones(shape=(1, latent_dimension))))[0], time, show=True, save=False, save_path='./results/sine_ones' + save_desc)
