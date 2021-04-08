import os
from math import sqrt
from random import randint
from statistics import mean  # , pstdev

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import (BatchNormalization, Dense,
                                     GlobalAveragePooling1D, GlobalMaxPool1D,
                                     LeakyReLU)
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from custom_losses import (DiscriminatorWassersteinLoss,
                           GeneratorWassersteinLoss)
from keras_data import data_to_dataset, plot_data
from keras_gan import GAN, WGAN, cWGAN
from keras_model_functions import plot_recurrence


def generate_sine(start, end, points, amplitude=1, frequency=1):
    time = np.linspace(start, end, points)
    signal = amplitude*np.sin(2*np.pi*frequency*time)
    return signal

def plot_sine(data, time, show=False, save=False, save_path='', plot_trend=False, trend_signal=None):
    plot_data(time,
     data, trend_signal if plot_trend and trend_signal is not None and len(trend_signal) > 0 else None,
      show=show, save=save, save_path=save_path)

def generate_image_summary(generator, latent_dim_size, time, num_rand_images=1, plot_trend=False, trend_signal=None, show=False, save=False, save_dir='./', save_desc=''):
    get_data = lambda gen, input_data: gen.predict(input_data(shape=(1, latent_dim_size)))[0]
    plot_sine(get_data(generator, tf.zeros), time, show=show, save=save, save_path=os.path.join( save_dir, 'sine_zeros' + save_desc),
     plot_trend=plot_trend, trend_signal=trend_signal)
    for idx in range(num_rand_images):
        plot_sine(get_data(generator, tf.random.normal), time, show=True, save=save, save_path=os.path.join( save_dir, 'sine_norm{}_{}'.format(save_desc, str(idx))),
         plot_trend=plot_trend, trend_signal=trend_signal)
    plot_sine(get_data(generator, tf.ones), time, show=show, save=save, save_path=os.path.join( save_dir, 'sine_ones' + save_desc),
     plot_trend=plot_trend, trend_signal=trend_signal)

def generate_conditional_image_summary(generator, conditional_data, latent_dim_size, time, num_rand_images=1, plot_trend=False, trend_signal=None, show=False, save=False, save_dir='./', save_desc=''):
    get_data = lambda gen, input_data, cond: gen.predict((input_data(shape=(1, latent_dim_size)), cond))[0]
    plot_sine(get_data(generator, tf.zeros, conditional_data), time, show=show, save=save, save_path=os.path.join( save_dir, 'sine_zeros' + save_desc),
     plot_trend=plot_trend, trend_signal=trend_signal)
    for idx in range(num_rand_images):
        plot_sine(get_data(generator, tf.random.normal, conditional_data), time, show=True, save=save, save_path=os.path.join( save_dir, 'sine_norm{}_{}'.format(save_desc, str(idx))),
         plot_trend=plot_trend, trend_signal=trend_signal)
    plot_sine(get_data(generator, tf.ones, conditional_data), time, show=show, save=save, save_path=os.path.join( save_dir, 'sine_ones' + save_desc),
     plot_trend=plot_trend, trend_signal=trend_signal)

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
    # linux = False
    # if linux:
    #     from tensorflow.compat.v1 import ConfigProto
    #     from tensorflow.compat.v1 import InteractiveSession

    #     config = ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     session = InteractiveSession(config=config)
    start_point, end_point, vector_size = 0, 2, 6000
    conditional, discriminator_mode, generator_mode, verbosity = True, 'cnn', 'cnn', 2
    latent_dimension, epochs, data_size, batch_size, data_type = 256, 32, int(1e2), 1, 'float32'
    save_desc = '__latent_dimension_{}_epochs_{}_data_size_{}_batch_size_{}_type_cnn_cnn'.format(latent_dimension, epochs, data_size, batch_size)
    early_stop = EarlyStopping(monitor='g_loss', mode='min', min_delta=1e-8, verbose=1, patience=3)
    checkpoint = ModelCheckpoint(filepath='./tmp/checkpoint', save_weights_only=True)
    callback_list = [checkpoint] # [early_stop, checkpoint]
    benign_data = [generate_sine(start_point, end_point, vector_size, frequency=1) for _ in range(int(data_size))] # generate 100 points of sine wave
    # for idx in range(2):
    #     plot_sine(benign_data[idx], show=True)
    dataset = data_to_dataset(benign_data, dtype=data_type, batch_size=batch_size, shuffle=True)
    # create discriminator and generator
    if conditional:
        save_desc = '__review_conditional_latent_dimension_{}_epochs_{}_data_size_{}_batch_size_{}_type_cnn_cnn'.format(latent_dimension, epochs, data_size, batch_size)
        benign_data, labels = [], []
        for _ in range(int(data_size)):
            frequency = randint(1, 3)
            benign_data.append(generate_sine(start_point, end_point, vector_size, frequency=frequency))
            labels.append([frequency])
        # for idx in range(2):
        #     plot_sine(benign_data[idx], show=True)
        # dataset = data_to_dataset(benign_data, dtype=data_type, batch_size=batch_size, shuffle=True)
        benign_data, labels = np.array(benign_data), np.array(labels)
        num_frequencies = 4
        discriminator_input_label = keras.layers.Input(shape=(1,))
        discriminator_label_side = keras.layers.Embedding(num_frequencies, 50)(discriminator_input_label)
        discriminator_label_side = Dense(vector_size)(discriminator_label_side)
        discriminator_label_side = layers.Reshape((vector_size, 1))(discriminator_label_side)

        discriminator_vector_input = layers.Input((vector_size,))
        discriminator_vector_side = layers.Reshape((vector_size, 1))(discriminator_vector_input)

        discriminator = layers.Concatenate()([discriminator_label_side, discriminator_vector_side])
        discriminator = layers.Conv1D(16, 5, strides=3)(discriminator)
        # discriminator = layers.BatchNormalization()(discriminator)
        discriminator = layers.LeakyReLU(alpha=0.2, dtype=data_type)(discriminator)
        discriminator = layers.Conv1D(8, (3), strides=(3), dtype=data_type)(discriminator)
        # discriminator = layers.BatchNormalization()(discriminator)
        discriminator = layers.LeakyReLU(alpha=0.2, dtype=data_type)(discriminator)
        discriminator = layers.Conv1D(8, (3), dtype=data_type)(discriminator)
        # discriminator = layers.BatchNormalization()(discriminator)
        discriminator = layers.LeakyReLU(alpha=0.2, dtype=data_type)(discriminator)
        discriminator = layers.Conv1D(1, (3), dtype=data_type)(discriminator)
        # discriminator = layers.BatchNormalization()(discriminator)
        discriminator = layers.LeakyReLU(alpha=0.2, dtype=data_type)(discriminator)
        # discriminator = layers.GlobalAveragePooling1D()(discriminator)
        # discriminator = layers.GlobalMaxPooling1D()(discriminator)
        discriminator = layers.Flatten()(discriminator)
        discriminator = layers.Dense(64)(discriminator)
        discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
        discriminator = layers.Dense(32)(discriminator)
        discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
        discriminator = layers.Dense(1, dtype=data_type)(discriminator)

        generator_input_label = keras.layers.Input(shape=(1,))
        generator_label_side = keras.layers.Embedding(num_frequencies, 50)(generator_input_label)
        generator_label_side = layers.Dense(latent_dimension)(generator_label_side)
        generator_label_side = layers.LeakyReLU(alpha=0.2)(generator_label_side)
        generator_label_side = layers.Reshape((latent_dimension, 1))(generator_label_side)

        generator_vector_input = layers.Input((latent_dimension,))
        generator_vector_side = layers.Reshape((latent_dimension, 1), dtype=data_type)(generator_vector_input)

        generator = layers.Concatenate()([generator_label_side, generator_vector_side])
        generator = layers.Conv1DTranspose(16, 3, strides=3, dtype=data_type)(generator)
        generator = layers.BatchNormalization()(generator)
        generator = layers.LeakyReLU(alpha=0.2, dtype=data_type)(generator)
        generator = layers.Conv1DTranspose(16, 3, strides=3, dtype=data_type)(generator)
        generator = layers.BatchNormalization()(generator)
        generator = layers.LeakyReLU(alpha=0.2, dtype=data_type)(generator)
        generator = layers.Conv1DTranspose(16, 3, strides=3, dtype=data_type)(generator)
        generator = layers.BatchNormalization()(generator)
        generator = layers.LeakyReLU(alpha=0.2, dtype=data_type)(generator)
        generator = layers.Conv1D(1, (3), dtype=data_type)(generator)
        generator = layers.LeakyReLU(alpha=0.2, dtype=data_type)(generator)
        # generator = layers.BatchNormalization()(generator)
        # generator = layers.Reshape((vector_size,))(generator)
        # generator = layers.Reshape((4 * vector_size,))(generator)
        generator = layers.Flatten()(generator)
        generator = layers.Dense(32, activation=tf.cos, dtype=data_type)(generator)
        # generator = layers.BatchNormalization()(generator)
        generator = layers.Dense(vector_size, activation='tanh', dtype=data_type)(generator)
        discriminator = keras.Model(inputs=(discriminator_vector_input, discriminator_input_label), outputs=discriminator, name='conditional_discriminator')
        generator = keras.Model(inputs=(generator_vector_input, generator_input_label), outputs = generator, name='conditional_generator')
        discriminator.summary()
        generator.summary()
        cwgan = cWGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dimension)
        cwgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                    g_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                    # g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                    # d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
                    # g_loss_fn=GeneratorWassersteinLoss(),
                    # d_loss_fn=DiscriminatorWassersteinLoss()
        )
        cwgan.set_train_epochs(5, 1)
        generator.predict((tf.zeros(shape=(1, latent_dimension)), tf.constant([1])))
        cwgan.fit(x=benign_data, y=labels, epochs=epochs, batch_size=batch_size, callbacks=callback_list)
        generator.save('./models/conditional_sine_generator')
        discriminator.save('./models/conditional_sine_discriminator')
        time = np.linspace(start_point, end_point, num=vector_size)
        from pyts.image import RecurrencePlot
        rp, trend = RecurrencePlot(threshold='point', percentage=20), generate_sine(start_point, end_point, vector_size, amplitude=1, frequency=1)
        plot_recurrence(np.linspace(start_point, end_point, vector_size), generator.predict((tf.zeros(shape=(1, latent_dimension)), tf.constant([1])))[0], rp, show=True, save=False)
        plot_sine(generator.predict([tf.zeros(shape=(1, latent_dimension)), tf.constant([1])])[0], time, show=True, save=False, save_path='./results/sine_zeros' + save_desc)
        for idx in range(3):
            plot_sine(generator.predict((tf.random.normal(shape=(1, latent_dimension)), tf.constant([idx + 1])))[0], time, show=True, save=False, save_path='./results/sine_norm' + save_desc + '_' + str(idx))
        plot_sine(generator.predict((tf.ones(shape=(1, latent_dimension)), tf.constant([1])))[0], time, show=True, save=False, save_path='./results/sine_ones' + save_desc)
        generate_conditional_image_summary(generator=generator, conditional_data=tf.constant([1]), time=time, latent_dim_size=latent_dimension, num_rand_images=3, show=True, save=False,
            save_dir='./results', save_desc=save_desc, plot_trend=True, trend_signal=trend)
        exit()
    else:
        if discriminator_mode == 'cnn':
            discrim = keras.Sequential(
            [
                layers.Reshape((vector_size, 1,), input_shape=(vector_size,), dtype=data_type),
                layers.Conv1D(16, (5), strides=(3), dtype=data_type),
                # layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2, dtype=data_type),
                layers.Conv1D(8, (3), strides=(3), dtype=data_type),
                # layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2, dtype=data_type),
                layers.Conv1D(8, (3), dtype=data_type),
                # layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2, dtype=data_type),
                # layers.GlobalAveragePooling1D(),
                # layers.GlobalMaxPooling1D(),
                layers.Flatten(),
                layers.Dense(64),
                layers.LeakyReLU(alpha=0.2),
                layers.Dense(32),
                layers.LeakyReLU(alpha=0.2),
                layers.Dense(1, dtype=data_type) #, activation='sigmoid'),
            ],
            name="discriminator",
            )
        else:
            discrim = keras.Sequential([
                layers.Dense(vector_size, input_shape=(vector_size,), dtype=data_type),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2, dtype=data_type),
                layers.Dense(32, dtype=data_type),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2, dtype=data_type),
                layers.Dense(32, dtype=data_type),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2, dtype=data_type),
                layers.Dense(32, dtype=data_type),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2, dtype=data_type),
                layers.Dense(32, dtype=data_type),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2, dtype=data_type),
                layers.Dense(1, dtype=data_type)
            ],
            name="discriminator",
            )
        discrim.summary()
        print('Discriminator data type: '+discrim.dtype)
        if generator_mode == 'cnn':
            generator = keras.Sequential(
                [
                    # layers.Dense(vector_size * latent_dimension, input_shape=(latent_dimension,)),
                    # layers.LeakyReLU(alpha=0.2),
                    # layers.Dense(latent_dimension),
                    # layers.LeakyReLU(alpha=0.2),
                    layers.Reshape((latent_dimension, 1), input_shape=(latent_dimension,), dtype=data_type),
                    layers.Conv1DTranspose(16, 3, strides=3, dtype=data_type),
                    layers.BatchNormalization(),
                    layers.LeakyReLU(alpha=0.2, dtype=data_type),
                    layers.Conv1DTranspose(16, 3, strides=3, dtype=data_type),
                    layers.BatchNormalization(),
                    layers.LeakyReLU(alpha=0.2, dtype=data_type),
                    layers.Conv1DTranspose(16, 3, strides=3, dtype=data_type),
                    layers.BatchNormalization(),
                    layers.LeakyReLU(alpha=0.2, dtype=data_type),
                    layers.Conv1D(1, (3), dtype=data_type),
                    layers.LeakyReLU(alpha=0.2, dtype=data_type),
                    # layers.BatchNormalization(),
                    # layers.Reshape((vector_size,)),
                    # layers.Reshape((4 * vector_size,)),
                    layers.Flatten(),
                    layers.Dense(32, activation=tf.cos, dtype=data_type),
                    # layers.BatchNormalization(),
                    layers.Dense(vector_size, activation='tanh', dtype=data_type)
                ],
                name="generator",
            )
        else:
            generator = keras.Sequential(
                [
                    layers.Dense(2 * latent_dimension, input_shape=(latent_dimension,), dtype=data_type),
                    layers.BatchNormalization(),
                    layers.LeakyReLU(alpha=0.2, dtype=data_type),
                    layers.Dense(32, dtype=data_type),
                    layers.BatchNormalization(),
                    layers.LeakyReLU(alpha=0.2, dtype=data_type),
                    layers.Dense(32, dtype=data_type),
                    layers.BatchNormalization(),
                    layers.LeakyReLU(alpha=0.2, dtype=data_type),
                    layers.Dense(32, dtype=data_type),
                    layers.BatchNormalization(),
                    layers.Dense(32, activation=tf.cos, dtype=data_type),
                    layers.Dense(vector_size, activation='tanh', dtype=data_type)
                ],
                name="generator",
            )
        generator.summary()
    # generator attempts to produce even numbers, discriminator will tell if true or not
    wgan = WGAN(discriminator=discrim, generator=generator, latent_dim=latent_dimension)
    wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                g_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                # g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                # d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
                g_loss_fn=GeneratorWassersteinLoss(),
                d_loss_fn=DiscriminatorWassersteinLoss()
    )
    wgan.set_train_epochs(5, 1)
    wgan.fit(dataset, epochs=epochs, batch_size=batch_size, callbacks=callback_list)
    # generator.save('./models/sine_generator')
    # discrim.save('./models/sine_discriminator')
    time = np.linspace(start_point, end_point, num=vector_size)
    from pyts.image import RecurrencePlot
    rp, trend = RecurrencePlot(threshold='point', percentage=20), generate_sine(start_point, end_point, vector_size, amplitude=1, frequency=1)
    plot_recurrence(np.linspace(start_point, end_point, vector_size), generator.predict(tf.zeros(shape=(1, latent_dimension)))[0], rp, show=True, save=False)
    generate_image_summary(generator=generator, time=time, latent_dim_size=latent_dimension, num_rand_images=3, show=True, save=False,
     save_dir='./results', save_desc=save_desc, plot_trend=True, trend_signal=trend)
    # plot_sine(generator.predict(tf.zeros(shape=(1, latent_dimension)))[0], show=True, save=False, save_path='./results/sine_zeros' + save_desc)
    # plot_sine(standardize(generator.predict(tf.zeros(shape=(1, latent_dimension)))[0]), show=True, save=False)
    # for idx in range(3):
    #     plot_sine(generator.predict(tf.random.normal(shape=(1, latent_dimension)))[0], show=True, save=False, save_path='./results/sine_norm' + save_desc + '_' + str(idx))
    # plot_sine(generator.predict(tf.ones(shape=(1, latent_dimension)))[0], show=True, save=False, save_path='./results/sine_ones' + save_desc)
