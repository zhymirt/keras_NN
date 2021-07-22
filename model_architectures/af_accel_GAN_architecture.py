import tensorflow.keras as keras
from tensorflow import cos as cos
from tensorflow.keras import layers

from keras_gan import Autoencoder


def make_af_accel_discriminator(vector_size, summary=False, data_type='float32'):
    discriminator = keras.Sequential(
        [
            layers.Reshape((vector_size, 1), input_shape=(vector_size,)),
            layers.Conv1D(8, 5, strides=5, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            # layers.Conv1D(64, 5, strides=5, padding='same'),
            # layers.LeakyReLU(alpha=0.2),
            layers.Conv1D(8, 3, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1D(16, 3, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1D(32, 3, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1D(64, 3, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1D(64, 3, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1D(128, 3, strides=2, padding='valid'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1D(128, 3, strides=2, padding='valid'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1D(256, 3, 1, padding='valid'),
            layers.LeakyReLU(alpha=0.2),
            # layers.Conv1D(512, 3, 1, padding='valid'),
            # layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dense(1)
        ],
        name="discriminator",
    )
    if summary:
        discriminator.summary()
    return discriminator


def make_af_accel_fcc_generator(latent_dimension, data_size, summary=False, data_type='float32'):
    model = keras.Sequential([
        layers.Dense(latent_dimension, input_shape=(latent_dimension,), dtype=data_type),
        layers.BatchNormalization(),
        layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Dense(64, activation=cos),
        layers.Dense(data_size, activation='tanh', dtype=data_type)
        # layers.Dense(vector_size, dtype=data_type),
        # layers.Dense(vector_size, dtype=data_type)
    ],
        name="generator",
    )
    if summary:
        print(model.input_shape)
        print(model.summary())
        print(model.output_shape)
    return model


def make_af_accel_generator(latent_dimension, data_size, summary=False, data_type='float32'):
    mini_data, channels = 15, 64
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
            layers.Conv1DTranspose(1, 5, strides=5, padding='same', activation='tanh', dtype=data_type),
            layers.Reshape((data_size,))
        ],
        name="generator",
    )
    if summary:
        generator.summary()
    return generator


def make_conditional_af_accel_discriminator(vector_size, num_frequencies, summary=False, data_type='float32'):
    discriminator_input_label = keras.layers.Input(shape=(num_frequencies,))
    # discriminator_label_side = keras.layers.Embedding(num_frequencies, 50)(discriminator_input_label)
    discriminator_label_side = layers.Dense(vector_size)(discriminator_input_label)
    discriminator_label_side = layers.Reshape((vector_size, 1))(discriminator_label_side)

    discriminator_vector_input = layers.Input((vector_size,))
    discriminator_vector_side = layers.Reshape((vector_size, 1))(discriminator_vector_input)

    discriminator = layers.Concatenate()([discriminator_label_side, discriminator_vector_side])
    discriminator = layers.Conv1D(32, 5, strides=5, padding='same')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Conv1D(32, 5, strides=5, padding='same')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    # discriminator = layers.Conv1D(32, 3, strides=2, padding='same')(discriminator)
    # discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Conv1D(32, 3, strides=2, padding='same')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Conv1D(32, 3, strides=2, padding='same')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Conv1D(64, 3, strides=2, padding='same')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Conv1D(128, 3, strides=2, padding='valid')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    # discriminator = layers.Conv1D(64, 3, strides=1, padding='valid')(discriminator)
    # discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    # discriminator = layers.Conv1D(64, 3, strides=1, padding='valid')(discriminator)
    # discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Flatten()(discriminator)
    discriminator = layers.Dense(1, dtype=data_type)(discriminator)

    discriminator = keras.Model(inputs=(discriminator_vector_input, discriminator_input_label),
                                outputs=discriminator, name='conditional_discriminator')
    if summary:
        print(discriminator.input_shape)
        print(discriminator.summary())
        print(discriminator.output_shape)
    return discriminator


def make_conditional_af_accel_generator(latent_dimension, data_size, num_frequencies, summary=False,
                                        data_type='float32'):
    mini_data, channels = 12, 32
    generator_input_label = keras.layers.Input(shape=(num_frequencies,))
    # generator_label_side = keras.layers.Embedding(num_frequencies, 50)(generator_input_label)
    generator_label_side = layers.Dense(mini_data * channels)(generator_input_label)
    generator_label_side = layers.ReLU()(generator_label_side)
    generator_label_side = layers.Reshape((mini_data, channels))(generator_label_side)

    generator_vector_input = layers.Input((latent_dimension,))
    generator_vector_side = layers.Dense(mini_data * channels)(generator_vector_input)
    generator_vector_side = layers.Reshape((mini_data, channels))(generator_vector_side)

    generator = layers.Concatenate()([generator_label_side, generator_vector_side])
    generator = layers.Conv1DTranspose(64, 3, strides=2, padding='valid')(generator)
    generator = layers.BatchNormalization()(generator)
    generator = layers.ReLU()(generator)
    generator = layers.Conv1DTranspose(64, 3, strides=2, padding='same')(generator)
    generator = layers.BatchNormalization()(generator)
    generator = layers.ReLU()(generator)
    generator = layers.Conv1DTranspose(32, 3, strides=2, padding='same')(generator)
    generator = layers.BatchNormalization()(generator)
    generator = layers.ReLU()(generator)
    generator = layers.Conv1DTranspose(32, 3, strides=2, padding='same')(generator)
    generator = layers.BatchNormalization()(generator)
    generator = layers.ReLU()(generator)
    generator = layers.Conv1DTranspose(32, 5, strides=5, padding='same', activation=cos)(generator)
    generator = layers.BatchNormalization()(generator)
    generator = layers.Conv1DTranspose(1, 5, strides=5, padding='same', activation='tanh', dtype=data_type)(generator)
    generator = layers.Reshape((data_size,))(generator)

    generator = keras.Model(inputs=(generator_vector_input, generator_input_label), outputs=generator,
                            name='conditional_generator')
    if summary:
        print(generator.input_shape)
        print(generator.summary())
        print(generator.output_shape)
    return generator


def make_fcc_autoencoder(vector_size, latent_dimension, summary=False, data_type='float32'):
    encoder = keras.Sequential([
        # layers.Dense(256, 'relu', input_shape=(vector_size,)),
        # layers.Dense(256, 'relu', dtype=data_type),
        # layers.Dense(128, 'relu', dtype=data_type),
        # layers.Dense(latent_dimension, 'tanh', dtype=data_type)

        layers.Dense(latent_dimension, activation='relu', input_shape=(vector_size,))
    ])
    decoder = keras.Sequential([
        # layers.Dense(128, 'relu', input_shape=(latent_dimension,)),
        # layers.Dense(256, 'relu', dtype=data_type),
        # layers.Dense(256, 'relu', dtype=data_type),
        # layers.Dense(vector_size, 'tanh', dtype=data_type)

        layers.Dense(vector_size, activation='tanh', input_shape=(latent_dimension,))
    ])
    model = Autoencoder(encoder=encoder, decoder=decoder, latent_dimension=latent_dimension)
    if summary:
        print(model.summary())
    return model


def make_cnn_autoencoder(vector_size, latent_dimension, summary=False, data_type='float32'):
    encoder = keras.Sequential([
        layers.Reshape((vector_size, 1), input_shape=(vector_size,)),
        layers.Conv1D(32, 3, strides=2, padding='same', activation='relu'),
        layers.Conv1D(32, 3, strides=2, padding='same', activation='relu'),
        layers.Conv1D(1, 3, strides=2, padding='same', activation='relu')
    ])
    decoder = keras.Sequential([
        layers.Conv1DTranspose(32, 3, strides=2, padding='same', activation='relu'),
        layers.Conv1DTranspose(32, 3, strides=2, padding='same', activation='relu'),
        layers.Conv1DTranspose(1, 3, strides=2, padding='same', activation='tanh'),
        layers.Reshape((vector_size,))
    ])
    model = Autoencoder(encoder=encoder, decoder=decoder, latent_dimension=latent_dimension)
    if summary:
        print(encoder.summary())
        print(decoder.summary())
    return model
