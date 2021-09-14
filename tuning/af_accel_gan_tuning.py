import os

import kerastuner as kt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras, cos

from AF_gan import metric_fft_score, normalize_data
from af_accel_GAN import load_data_files, prepare_data
from keras_data import data_to_dataset
from keras_gan import WGAN
from model_architectures.af_accel_GAN_architecture import make_af_accel_discriminator, make_af_accel_generator


def model_builder(hp):
    latent_dimension, vector_size = 256, 5000
    # d_hp_units = [hp.Int('d_units_{}'.format(idx), min_value=8, max_value=128, step=8) for idx in range(10)]
    # d_hp_units = [hp.Choice('d_unit_choice_{}'.format(idx), values=[8, 16, 32, 64, 128], default=32) for idx in range(10)]
    # discriminator = keras.Sequential(
    # [
    #     layers.Reshape((vector_size, 1), input_shape=(vector_size,)),
    #     layers.Conv1D(32, 5, strides=5, padding='same'),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Conv1D(32, 5, strides=5, padding='same'),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Conv1D(32, 3, strides=2, padding='same'),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Conv1D(32, 3, strides=2, padding='same'),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Conv1D(64, 3, strides=2, padding='same'),
    #     layers.LeakyReLU(alpha=0.2),
    #     # layers.Conv1D(32, 3, strides=2, padding='same'),
    #     # layers.LeakyReLU(alpha=0.2),
    #     layers.Conv1D(64, 3, strides=2, padding='same'),
    #     layers.LeakyReLU(alpha=0.2),
    #     # layers.Conv1D(64, 3, 1, padding='valid'),
    #     # layers.LeakyReLU(alpha=0.2),
    #     # layers.Conv1D(256, 3, 1, padding='valid'),
    #     # layers.LeakyReLU(alpha=0.2),
    #     layers.Flatten(),
    #     layers.Dense(1)
    # ],
    # name="discriminator",
    # )
    discriminator = make_af_accel_discriminator(vector_size)
    # g_hp_units = [hp.Int('g_units_{}'.format(idx), min_value=8, max_value=128, step=8) for idx in range(7)]
    # g_hp_units = [hp.Choice('g_unit_choice_{}'.format(idx), values=[8, 16, 32, 64, 128], default=32) for idx in range(7)]
    mini_data, channels = 15, hp.Choice('g_channels', values=[8, 16, 32, 64])
    generator = keras.Sequential(
        [
            layers.Dense(mini_data * channels, input_shape=(latent_dimension,)),
            layers.Reshape((mini_data, channels)),
            layers.Conv1DTranspose(64, 1, strides=1, padding='same'),
            # layers.BatchNormalization(),
            layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(64, 3, strides=1, padding='same'),
            # layers.BatchNormalization(),
            layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(64, 3, strides=2, padding='valid'),
            # layers.BatchNormalization(),
            layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(32, 3, strides=2, padding='same'),
            # layers.BatchNormalization(),
            layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(16, 3, strides=2, padding='valid'),
            # layers.BatchNormalization(),
            layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(8, 3, strides=2, padding='same'),
            # layers.BatchNormalization(),
            layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(4, 3, strides=2, padding='same'),
            # layers.BatchNormalization(),
            layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
            # layers.Conv1DTranspose(32, 5, strides=5, padding='same', activation=cos),
            # layers.BatchNormalization(),
            layers.Conv1DTranspose(2, 3, strides=2, padding='same', activation=cos),
            # layers.BatchNormalization(),
            layers.Conv1DTranspose(1, 5, strides=5, padding='same', activation='tanh', dtype='float32'),
            layers.Reshape((vector_size,))
        ],
        name="generator",
    )
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
                         max_epochs=32,
                         factor=3,
                         directory='keras_tuning',
                         project_name='af_accel_tuning'
                         )
    stop_early = tf.keras.callbacks.EarlyStopping(monitor=obj_metric, patience=5)
    tuner.search(training_data, epochs=2, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps, tuner


if __name__ == '__main__':
    complete_data = load_data_files([os.path.join('../../acceleration_data', name) for name in ('accel_1.csv',
                                                                                             'accel_2.csv',
                                                                                             'accel_3.csv',
                                                                                             'accel_4.csv')],
                                    separate_time=False)
    # full_time = complete_data[:, :, 0]
    # full_data, labels = [], []
    # for example_set in complete_data.transpose((0, 2, 1)):
    #     for test_num, test in enumerate(example_set[1:]):
    #         labels.append(test_num + 1)
    #         full_data.append(test)
    # full_data, labels = np.array(full_data), np.array(labels)
    # data_size = full_data.shape[1]
    # normalized, scalars = normalize_data(full_data)
    prepared_data = prepare_data(complete_data, scaling='normalize', return_labels=False)
    normalized = prepared_data['normalized'].repeat(1e3, axis=0)  # 1e4
    dataset = data_to_dataset(normalized, dtype='float32', batch_size=16, shuffle=True)
    # Train
    best_hps, tuner = tune_model(model_builder, 'metric_fft_score', dataset)
    model = tuner.hypermodel.build(best_hps)
    print(dir(best_hps))
    print(best_hps.values)
    print("Discriminator and Generator learning rates: {}, {}".format(best_hps.get('d_learning_rate'),
                                                                      best_hps.get('g_learning_rate')))
    model.generator.summary()
    exit()
    history = model.fit(dataset)
    # for key in best_hps:
    #     print('Best {}: {}'.format(key, best_hps[key]))
