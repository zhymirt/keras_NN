import os

import keras_tuner as kt
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers
from tensorflow import keras, cos

from AF_gan import metric_fft_score, normalize_data
from af_accel_GAN import load_data_files, prepare_data
from custom_functions.custom_classes import WGAN, CWGAN
from model_architectures.af_accel_GAN_architecture import make_af_accel_discriminator


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


def conditional_model_builder(hp: kt.HyperParameters):
    vector_size, num_frequencies = 5000, 4
    latent_dimension = hp.Int("latent_dimension_size", 8, 256, 8)
    critic_num_layers = hp.Int("critic_num_layers", 1, 8)
    generator_num_layers = hp.Int("generator_num_layers", 0, 7)
    padding_type = [
        'same', 'same', 'same', 'valid', 'valid', 'same', 'valid', 'valid']
    # num_filters = [
    #     16, 32, 64, 128, 256, 512, 1024, 1024
    # ]
    discriminator_input_label = layers.Input(shape=(num_frequencies,))
    # discriminator_label_side = keras.layers.Embedding(num_frequencies, 50)(discriminator_input_label)
    # discriminator_label_side = layers.Dense(1)(discriminator_input_label)
    # discriminator_label_side = layers.Reshape((vector_size, 1))(discriminator_label_side)

    discriminator_vector_input = layers.Input((vector_size,))
    # discriminator_vector_side = layers.Reshape((vector_size, 1))(discriminator_vector_input)

    # discriminator = layers.Concatenate()([discriminator_label_side, discriminator_vector_input])
    discriminator = layers.Reshape((vector_size, 1))(discriminator_vector_input)
    # discriminator = layers.Concatenate()([discriminator_label_side, discriminator_vector_side])
    discriminator = layers.Conv1D(8, 5, strides=5, padding='same')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    # discriminator = layers.Conv1D(32, 5, strides=5, padding='same')(discriminator)
    # discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    for idx in range(critic_num_layers):
        discriminator = layers.Conv1D(
            1 << 3 + idx,  # 8 * 2**idx
            3, strides=2,
            padding=padding_type[idx])(discriminator)
        discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    # discriminator = layers.Conv1D(16, 3, strides=2, padding='same')(discriminator)
    # discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    # discriminator = layers.Conv1D(32, 3, strides=2, padding='same')(discriminator)
    # discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    # discriminator = layers.Conv1D(64, 3, strides=2, padding='same')(discriminator)
    # discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    # discriminator = layers.Conv1D(128, 3, strides=2, padding='valid')(discriminator)
    # discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    # discriminator = layers.Conv1D(256, 3, strides=2, padding='valid')(discriminator)
    # discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    # discriminator = layers.Conv1D(512, 3, strides=2, padding='same')(discriminator)
    # discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    # discriminator = layers.Conv1D(1024, 3, strides=2, padding='valid')(discriminator)
    # discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    # discriminator = layers.Conv1D(1024, 3, strides=2, padding='valid')(discriminator)
    # discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Flatten()(discriminator)
    discriminator = layers.Concatenate()([discriminator, discriminator_input_label])
    discriminator = layers.Dense(1)(discriminator)

    discriminator = keras.Model(inputs=(discriminator_vector_input, discriminator_input_label),
                                outputs=discriminator, name='conditional_discriminator')
    # Generator Model
    mini_data_options = (7, 15, 31, 62, 125, 250, 500, 1_000)
    padding_options = (
        'valid', 'valid', 'same', 'valid', 'same', 'same', 'same')
    mini_data = mini_data_options[
        len(mini_data_options) - 1 - generator_num_layers]
    channels = 32 if generator_num_layers == 0 else 1 << (
            generator_num_layers + 2)
    # 2**( generator_num_layers + 2)
    # mini_data = 15  # hp.Choice('mini_data_size', values=[15 * 2**n for n in range(6)])
    # channels = 512  # hp.Choice('g_channels', values=[8, 16, 32, 64])
    generator_input_label = keras.layers.Input(shape=(num_frequencies,))

    generator_vector_input = layers.Input((latent_dimension,))

    generator = layers.Concatenate()([generator_input_label, generator_vector_input])
    generator = layers.Dense(mini_data * channels)(generator)
    generator = layers.Reshape((mini_data, channels))(generator)
    # TODO correct code for generator
    for idx in range(generator_num_layers):
        generator = layers.Conv1D(
            # same as 8 * 2**n
            1 << 3 + idx,  # 8 * 2**idx,
            3, strides=2,
            padding=padding_options[idx])(generator)
        generator = layers.ReLU(alpha=0.2)(generator)
        # Assuming it'll be even, just bit shift one to divide by 2
        channels = channels >> 1  # int(channels/2)
    if generator_num_layers > 0:
        generator = layers.Conv1D()
    # generator = layers.Conv1DTranspose(256, 3, strides=2, padding='valid')(generator)
    # # generator = layers.BatchNormalization()(generator
    # generator = layers.ReLU()(generator)  # layers.LeakyReLU(alpha=0.2)(generator)
    # generator = layers.Conv1DTranspose(128, 3, strides=2, padding='same')(generator)
    # # generator = layers.BatchNormalization()(generator
    # generator = layers.ReLU()(generator)  # layers.LeakyReLU(alpha=0.2)(generator)
    # generator = layers.Conv1DTranspose(64, 3, strides=2, padding='valid')(generator)
    # # generator = layers.BatchNormalization()(generator
    # generator = layers.ReLU()(generator)  # layers.LeakyReLU(alpha=0.2)(generator)
    # generator = layers.Conv1DTranspose(32, 3, strides=2, padding='same')(generator)
    # # generator = layers.BatchNormalization()(generator
    # generator = layers.ReLU()(generator)  # layers.LeakyReLU(alpha=0.2)(generator)
    # generator = layers.Conv1DTranspose(16, 3, strides=2, padding='same')(generator)
    # # generator = layers.BatchNormalization()(generator
    # generator = layers.ReLU()(generator)  # layers.LeakyReLU(alpha=0.2)(generator)
    # # generator = layers.Conv1DTranspose(32, 5, strides=5, padding='same', activation=cos)(generator
    # # generator = layers.BatchNormalization()(generator
    # generator = layers.Conv1DTranspose(8, 3, strides=2, padding='same', activation=cos, use_bias=False)(generator)
    # # generator = layers.BatchNormalization()(generator
    generator = layers.Conv1DTranspose(1, 5, strides=5, padding='same', activation='tanh', use_bias=False)(generator)
    generator = layers.Reshape((vector_size,))(generator)

    generator = keras.Model(
        inputs=(generator_vector_input, generator_input_label),
        outputs=generator, name='conditional_generator')

    cwgan = CWGAN(
        discriminator=discriminator, generator=generator,
        latent_dim=latent_dimension)
    d_hp_lr = hp.Choice('d_learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    g_hp_lr = hp.Choice('g_learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    cwgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=d_hp_lr),
        g_optimizer=keras.optimizers.Adam(learning_rate=g_hp_lr),
        metrics=[metric_fft_score])
    return cwgan


def tune_model(model_fn, obj_metric, training_data):
    tuner = kt.Hyperband(model_fn,
                         objective=kt.Objective(obj_metric, direction="min"),
                         max_epochs=32,
                         factor=3,
                         directory='keras_tuning',
                         project_name='conditional_af_accel_tuning'
                         )
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor=obj_metric, patience=5)
    tuner.search(training_data, epochs=2, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps, tuner


def main():
    # Initial values
    folder = '../../acceleration_data'
    filenames = ('accel_1.csv', 'accel_2.csv', 'accel_3.csv', 'accel_4.csv')
    # Collect data
    complete_data = load_data_files(
        [os.path.join(folder, name) for name in filenames],
        separate_time=False)
    prepared_data = prepare_data(
        complete_data, scaling='normalize', return_labels=True)
    normalized = prepared_data['normalized'].repeat(4e3, axis=0)  # 1e4
    labels = prepared_data['labels'].repeat(4e3, axis=0)
    mlb = MultiLabelBinarizer()
    new_labels = mlb.fit_transform(labels)
    dataset = (normalized, new_labels)
    # dataset = data_to_dataset(
    #     normalized, dtype='float32', batch_size=16, shuffle=True)
    # Train
    best_hps, tuner = tune_model(
        conditional_model_builder, 'metric_fft_score', dataset)
    model = tuner.hypermodel.build(best_hps)
    print(dir(best_hps))
    print(best_hps.values)
    print("Discriminator and Generator learning rates: {}, {}".format(
        best_hps.get('d_learning_rate'), best_hps.get('g_learning_rate')))
    model.generator.summary()
    exit()
    history = model.fit(dataset)
    # for key in best_hps:
    #     print('Best {}: {}'.format(key, best_hps[key]))


if __name__ == '__main__':
    main()
