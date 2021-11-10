import os

import kerastuner as kt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras, cos

from AF_gan import metric_fft_score, normalize_data
from keras_data import data_to_dataset
from custom_classes import WGAN


def model_builder(hp):
    vector_size, num_data_types = 5501, 5
    latent_dimension = 256
    # d_hp_units = [hp.Int('d_units_{}'.format(idx), min_value=8, max_value=128, step=8) for idx in range(10)]
    d_hp_units = [hp.Choice('d_unit_choice_{}'.format(idx), values=[8, 16, 32, 64, 128], default=32) for idx in range(10)]
    discriminator = keras.Sequential(
    [
        layers.Reshape((vector_size, num_data_types), input_shape=(vector_size, num_data_types,)),
        # layers.Conv2D(32, (3, 1), strides=(2, 1), padding='valid'),
        # layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(d_hp_units[0], 5, strides=5, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(d_hp_units[1], 3, strides=3, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(d_hp_units[2], 3, strides=3, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(d_hp_units[3], 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(d_hp_units[4], 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(d_hp_units[5], 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(d_hp_units[6], 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(d_hp_units[7], 3, 1, padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(d_hp_units[8], 3, 1, padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(d_hp_units[9], 3, 1, padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dense(1)
    ])
    # g_hp_units = [hp.Int('g_units_{}'.format(idx), min_value=8, max_value=128, step=8) for idx in range(7)]
    g_hp_units = [hp.Choice('g_unit_choice_{}'.format(idx), values=[8, 16, 32, 64, 128], default=32) for idx in range(7)]
    mini_data, channels = 15, hp.Choice('g_channels', values=[8, 16, 32, 64])
    generator = keras.Sequential(
        [
            layers.Dense(mini_data * channels, input_shape=(latent_dimension,)),
            layers.Reshape((mini_data, channels)),
            layers.Conv1DTranspose(g_hp_units[0], 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(g_hp_units[1], 3, strides=2, padding='valid'),
            layers.BatchNormalization(),
            layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(g_hp_units[2], 3, strides=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(g_hp_units[3], 3, strides=2, padding='valid'),
            layers.BatchNormalization(),
            layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(g_hp_units[4], 3, strides=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1DTranspose(g_hp_units[5], 5, strides=5, padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(g_hp_units[6], 3, strides=1, padding='valid', activation=cos),
            layers.BatchNormalization(),
            layers.Conv1D(5, 3, strides=1, padding='valid', activation='tanh'),
            layers.Reshape((vector_size, num_data_types,))
            ])
    wgan = WGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dimension)
    d_hp_lr = hp.Choice('d_learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    g_hp_lr = hp.Choice('g_learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=d_hp_lr),
                 g_optimizer=keras.optimizers.Adam(learning_rate=g_hp_lr),
                 metrics=[metric_fft_score]
                 )
    return wgan

def tune_model(model_fn, obj_metric, training_data):
    tuner = kt.Hyperband(model_fn,
                         objective=kt.Objective(obj_metric, direction="min"),
                         max_epochs=100,
                         factor=3,
                         directory='keras_tuning',
                        project_name='AF_GAN_tuning'
                         )
    stop_early = tf.keras.callbacks.EarlyStopping(monitor=obj_metric, patience=5)
    tuner.search(training_data, epochs=50, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps, tuner

if __name__=='__main__':
    # Get data
    # time, benign_data = read_file_to_arrays('../signal_data/T04.txt')[0], [
    #     read_file_to_arrays(os.path.join('../signal_data', name))[1] for name in ['T04.txt',
    #                                                                               'T04repeat.txt', 'T05.txt', 'T06.txt',
    #                                                                               'T07.txt', 'T08.txt']]
    time, benign_data = np.load('../../signal_data/time_np.npy'), np.concatenate(
        [[np.load(os.path.join('../../signal_data', name + '_np.npy'))] for name in ['T04',
                                                                                  'T04repeat', 'T05', 'T06', 'T07',
                                                                                  'T08']])
    benign_data_transposed = np.transpose(benign_data, (0, 2, 1))
    transformed, scalars = normalize_data(benign_data_transposed)
    transformed = transformed.transpose((0, 2, 1))
    # Repeat data to increase data size
    transformed = transformed.repeat(2e3, axis=0)  # 1e4
    # benign_data = np.array(benign_data) # Training on one example to narrow down issue
    dataset = data_to_dataset(transformed, dtype='float32', batch_size=16, shuffle=True)
    # Train
    best_hps, tuner = tune_model(model_builder, 'metric_fft_score', dataset)
    for key in best_hps:
        print('Best {}: {}'.format(key, best_hps[key]))
