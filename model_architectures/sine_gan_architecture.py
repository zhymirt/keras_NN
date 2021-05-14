from tensorflow import cos, keras
from tensorflow.keras import layers


def make_sine_gan_fcc_discriminator(vector_size, summary=False, data_type='float32'):
    model = keras.Sequential([
        layers.Dense(32, input_shape=(vector_size,), dtype=data_type),
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
    if summary:
        print(model.input_shape)
        print(model.summary())
        print(model.output_shape)
    return model


def make_sine_gan_fcc_generator(latent_dimension, vector_size, summary=False, data_type='float32'):
    model = keras.Sequential([
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
        layers.Dense(32, activation=cos, dtype=data_type),
        layers.Dense(vector_size, activation='tanh', dtype=data_type)
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


def make_sine_gan_cnn_discriminator(vector_size, summary=False, data_type='float32'):
    model = keras.Sequential([
        layers.Reshape((vector_size, 1,), input_shape=(vector_size,), dtype=data_type),
        layers.Conv1D(64, 5, strides=5, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(32, 5, strides=5, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(64, 3, strides=3, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(32, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(64, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(64, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(64, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        # layers.Conv1D(64, 3, strides=2, padding='same'),
        # layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(256, 3, strides=1, padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        # layers.GlobalAveragePooling1D(),
        # layers.GlobalMaxPooling1D(),
        layers.Flatten(),
        # layers.Dense(32),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Dense(32),
        # layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, dtype=data_type)  # , activation='sigmoid'),
    ],
        name="discriminator",
    )
    if summary:
        print(model.input_shape)
        print(model.summary())
        print(model.output_shape)
    return model


def make_sine_gan_cnn_generator(latent_dimension, summary=False, data_type='float32'):
    mini_data, channels = 12, 64
    flattened = mini_data * channels
    model = keras.Sequential([
        # layers.UpSampling1D(),
        layers.Dense(flattened, input_shape=(latent_dimension,)),
        # layers.BatchNormalization(),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Dense(32),
        # layers.BatchNormalization(),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Dense(flattened),
        # layers.BatchNormalization(),
        # layers.LeakyReLU(alpha=0.2),
        layers.Reshape((mini_data, channels)),
        layers.Conv1DTranspose(32, 3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        # layers.Conv1DTranspose(64, 3, strides=2, padding='same'),
        # layers.BatchNormalization(),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv1DTranspose(64, 3, strides=2, padding='same'),
        # layers.BatchNormalization(),
        # layers.LeakyReLU(alpha=0.2),
        layers.Conv1DTranspose(32, 3, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1DTranspose(32, 3, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        # layers.Conv1DTranspose(16, 3, strides=3, padding='valid'),
        # layers.BatchNormalization(),
        # layers.LeakyReLU(alpha=0.2),
        layers.Conv1DTranspose(32, 5, strides=5, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        # layers.Conv1DTranspose(8, 3, strides=1, padding='same'),
        # layers.BatchNormalization(),
        # layers.LeakyReLU(alpha=0.2),
        layers.Conv1DTranspose(32, 5, strides=5, padding='same', activation=cos),
        layers.BatchNormalization(),
        # layers.LeakyReLU(alpha=0.2),
        layers.Conv1DTranspose(1, 5, strides=5, padding='same', activation='tanh', dtype=data_type),  # get signal stand
        # layers.BatchNormalization(),
        layers.Conv1DTranspose(1, 1, strides=1, padding='same', dtype=data_type, use_bias=False),  # choose amplitude
        # layers.BatchNormalization(),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv1DTranspose(1, 5, strides=1, padding='same'),
        layers.Flatten(),
        # layers.Dense(32, activation=tf.cos),
        # layers.BatchNormalization(),
        # layers.Dense(vector_size, activation='tanh', dtype=data_type)
    ],
        name="generator",
    )
    if summary:
        print(model.input_shape)
        print(model.summary())
        print(model.output_shape)
    return model


# num_frequencies = 4


def make_conditional_sine_gan_cnn_discriminator(vector_size, num_frequencies, summary=False, data_type='float32'):
    discriminator_input_label = keras.layers.Input(shape=(1,))
    discriminator_label_side = keras.layers.Embedding(num_frequencies, 50)(discriminator_input_label)
    discriminator_label_side = layers.Dense(vector_size)(discriminator_label_side)
    discriminator_label_side = layers.Reshape((vector_size, 1))(discriminator_label_side)

    discriminator_vector_input = layers.Input((vector_size,))
    discriminator_vector_side = layers.Reshape((vector_size, 1))(discriminator_vector_input)

    discriminator = layers.Concatenate()([discriminator_label_side, discriminator_vector_side])
    discriminator = layers.Conv1D(16, 5, strides=3)(discriminator)
    # discriminator = layers.BatchNormalization()(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2, dtype=data_type)(discriminator)
    discriminator = layers.Conv1D(8, 3, strides=3, dtype=data_type)(discriminator)
    # discriminator = layers.BatchNormalization()(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2, dtype=data_type)(discriminator)
    discriminator = layers.Conv1D(8, 3, dtype=data_type)(discriminator)
    # discriminator = layers.BatchNormalization()(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2, dtype=data_type)(discriminator)
    discriminator = layers.Conv1D(1, 3, dtype=data_type)(discriminator)
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

    discriminator = keras.Model(inputs=(discriminator_vector_input, discriminator_input_label),
                                outputs=discriminator, name='conditional_discriminator')
    if summary:
        print(discriminator.input_shape)
        print(discriminator.summary())
        print(discriminator.output_shape)
    return discriminator


def make_conditional_sine_gan_cnn_generator(latent_dimension, vector_size, num_frequencies, summary=False, data_type='float32'):
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
    generator = layers.Conv1D(1, 3, dtype=data_type)(generator)
    generator = layers.LeakyReLU(alpha=0.2, dtype=data_type)(generator)
    # generator = layers.BatchNormalization()(generator)
    # generator = layers.Reshape((vector_size,))(generator)
    # generator = layers.Reshape((4 * vector_size,))(generator)
    generator = layers.Flatten()(generator)
    generator = layers.Dense(32, activation=cos, dtype=data_type)(generator)
    # generator = layers.BatchNormalization()(generator)
    generator = layers.Dense(vector_size, dtype=data_type)(generator)

    generator = keras.Model(inputs=(generator_vector_input, generator_input_label), outputs=generator,
                            name='conditional_generator')
    if summary:
        print(generator.input_shape)
        print(generator.summary())
        print(generator.output_shape)
    return generator
