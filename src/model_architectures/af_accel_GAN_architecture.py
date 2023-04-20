from tensorflow import cos as cos
from tensorflow.keras import layers, Model, Sequential

from custom_functions.custom_classes import Autoencoder, VAE, Sampling


def make_af_accel_discriminator(vector_size, summary=False, data_type='float32') -> Model:
    discriminator = Sequential(
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
            layers.Dense(1, dtype=data_type)
        ],
        name="discriminator",
    )
    if summary:
        discriminator.summary()
    return discriminator


def make_af_accel_fcc_generator(
        latent_dimension, data_size, summary=False, data_type='float32') -> Model:
    model = Sequential([
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


def make_af_accel_generator(
        latent_dimension, data_size, summary=False, data_type='float32') -> Model:
    mini_data, channels = 15, 32
    generator = Sequential(
        [
            layers.Dense(mini_data * channels, input_shape=(latent_dimension,)),
            layers.Reshape((mini_data, channels)),
            layers.Conv1DTranspose(32, 1, strides=1, padding='same'),
            # layers.BatchNormalization(),
            layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(32, 3, strides=1, padding='same'),
            # layers.BatchNormalization(),
            layers.ReLU(),  # layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(32, 3, strides=2, padding='valid'),
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


def make_conditional_af_accel_discriminator(
        vector_size, num_frequencies, summary=False, data_type='float32') -> Model:
    discriminator_input_label = layers.Input(shape=(num_frequencies,))
    # discriminator_label_side = layers.Embedding(num_frequencies, 50)(discriminator_input_label)
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
    discriminator = layers.Conv1D(16, 3, strides=2, padding='same')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Conv1D(32, 3, strides=2, padding='same')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Conv1D(64, 3, strides=2, padding='same')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Conv1D(128, 3, strides=2, padding='valid')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Conv1D(256, 3, strides=2, padding='valid')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Conv1D(512, 3, strides=2, padding='same')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    # discriminator = layers.Conv1D(1024, 3, strides=2, padding='valid')(discriminator)
    # discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    # discriminator = layers.Conv1D(1024, 3, strides=2, padding='valid')(discriminator)
    # discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Flatten()(discriminator)
    discriminator = layers.Concatenate()([discriminator, discriminator_input_label])
    discriminator = layers.Dense(1, dtype=data_type)(discriminator)

    discriminator = Model(inputs=(discriminator_vector_input, discriminator_input_label),
                                outputs=discriminator, name='conditional_discriminator')
    if summary:
        print(discriminator.input_shape)
        print(discriminator.summary())
        print(discriminator.output_shape)
    return discriminator


def make_conditional_af_accel_generator(
        latent_dimension, data_size, num_frequencies, summary=False,
                                        data_type='float32') -> Model:
    mini_data, channels = 62, 64  # 15, 512
    generator_input_label = layers.Input(shape=(num_frequencies,))
    # generator_label_side = layers.Dense(1)(generator_input_label)

    # generator_label_side = layers.Embedding(num_frequencies + 1, 50, input_length=num_frequencies)(generator_input_label)
    # generator_label_side = layers.Dense(mini_data * channels)(generator_input_label)
    # generator_label_side = layers.ReLU()(generator_label_side)
    # generator_label_side = layers.Reshape((mini_data, channels))(generator_label_side)

    generator_vector_input = layers.Input((latent_dimension,))
    # generator_vector_side = layers.Dense(mini_data * channels)(generator_vector_input)
    # generator_vector_side = layers.Reshape((mini_data, channels))(generator_vector_side)

    generator = layers.Concatenate()([generator_input_label, generator_vector_input])
    # Beginning of edits
    generator = layers.Dense(mini_data * channels)(generator)
    generator = layers.Reshape((mini_data, channels))(generator)




    # generator = layers.Conv1DTranspose(256, 3, strides=2, padding='valid')(generator)
    # # generator = layers.BatchNormalization()(generator
    # generator = layers.ReLU()(generator)  # layers.LeakyReLU(alpha=0.2)(generator)
    # generator = layers.Conv1DTranspose(128, 3, strides=2, padding='same')(generator)
    # # generator = layers.BatchNormalization()(generator

    # End of edits
    generator = layers.ReLU()(generator)  # layers.LeakyReLU(alpha=0.2)(generator)
    generator = layers.Conv1DTranspose(64, 3, strides=2, padding='valid')(generator)
    # generator = layers.BatchNormalization()(generator
    generator = layers.ReLU()(generator)  # layers.LeakyReLU(alpha=0.2)(generator)
    generator = layers.Conv1DTranspose(32, 3, strides=2, padding='same')(generator)
    # generator = layers.BatchNormalization()(generator
    generator = layers.ReLU()(generator)  # layers.LeakyReLU(alpha=0.2)(generator)
    generator = layers.Conv1DTranspose(16, 3, strides=2, padding='same')(generator)
    # generator = layers.BatchNormalization()(generator
    generator = layers.ReLU()(generator)  # layers.LeakyReLU(alpha=0.2)(generator)
    # generator = layers.Conv1DTranspose(32, 5, strides=5, padding='same', activation=cos)(generator
    # generator = layers.BatchNormalization()(generator
    generator = layers.Conv1DTranspose(8, 3, strides=2, padding='same', activation=cos, use_bias=False)(generator)
    # generator = layers.BatchNormalization()(generator
    generator = layers.Conv1DTranspose(1, 5, strides=5, padding='same', activation='tanh', use_bias=False, dtype=data_type)(generator)
    # generator = layers.Conv1DTranspose(64, 3, strides=2, padding='valid')(generator)
    # generator = layers.BatchNormalization()(generator)
    # generator = layers.ReLU()(generator)
    # generator = layers.Conv1DTranspose(64, 3, strides=2, padding='same')(generator)
    # generator = layers.BatchNormalization()(generator)
    # generator = layers.ReLU()(generator)
    # generator = layers.Conv1DTranspose(32, 3, strides=2, padding='same')(generator)
    # generator = layers.BatchNormalization()(generator)
    # generator = layers.ReLU()(generator)
    # generator = layers.Conv1DTranspose(32, 3, strides=2, padding='same')(generator)
    # generator = layers.BatchNormalization()(generator)
    # generator = layers.ReLU()(generator)
    # generator = layers.Conv1DTranspose(32, 5, strides=5, padding='same', activation=cos)(generator)
    # generator = layers.BatchNormalization()(generator)
    # generator = layers.Conv1DTranspose(1, 5, strides=5, padding='same', activation='tanh', dtype=data_type)(generator)
    generator = layers.Reshape((data_size,))(generator)

    generator = Model(inputs=(generator_vector_input, generator_input_label), outputs=generator,
                            name='conditional_generator')
    if summary:
        print(generator.input_shape)
        print(generator.summary())
        print(generator.output_shape)
    return generator


def make_fcc_autoencoder(
        vector_size, latent_dimension, summary=False, data_type='float32') -> Model:
    encoder = Sequential([
        layers.Dense(4096, 'relu', input_shape=(vector_size,)),
        layers.Dense(2048, 'relu', dtype=data_type),
        layers.Dense(512, activation=cos, dtype=data_type),
        layers.Dense(latent_dimension, 'tanh', dtype=data_type)

        # layers.Dense(latent_dimension, activation='relu', input_shape=(vector_size,))
    ])
    decoder = Sequential([
        layers.Dense(512, 'relu', input_shape=(latent_dimension,)),
        layers.Dense(2048, 'relu', dtype=data_type),
        layers.Dense(4096, 'relu', dtype=data_type),
        layers.Dense(vector_size, 'tanh', dtype=data_type)

        # layers.Dense(vector_size, activation='tanh', input_shape=(latent_dimension,))
    ])
    model = Autoencoder(encoder=encoder, decoder=decoder, latent_dimension=latent_dimension)
    if summary:
        print(encoder.summary())
        print(decoder.summary())
    return model


def make_conditional_fcc_autoencoder(
        vector_size, latent_dimension, num_frequencies, summary=False,
        data_type='float32') -> Model:
    encoder_vector_input = layers.Input((vector_size,))
    encoder_label_input = layers.Input((num_frequencies,))

    encoder_input = layers.Concatenate()([encoder_label_input, encoder_vector_input])

    encoder = layers.Dense(4096, 'relu', input_shape=(vector_size,))(encoder_input)
    encoder = layers.Dense(2048, 'relu', dtype=data_type)(encoder)
    encoder = layers.Dense(512, activation=cos, dtype=data_type)(encoder)
    encoder = layers.Dense(latent_dimension, 'tanh', dtype=data_type)(encoder)

    encoder_model = Model(inputs=(encoder_vector_input, encoder_label_input), outputs=encoder, name='encoder')

    decoder_vector_input = layers.Input((latent_dimension,))
    decoder_label_input = layers.Input((num_frequencies,))

    decoder_input = layers.Concatenate()([decoder_label_input, decoder_vector_input])

    decoder = layers.Dense(512, 'relu', input_shape=(latent_dimension,))(decoder_input)
    decoder = layers.Dense(2048, 'relu', dtype=data_type)(decoder)
    decoder = layers.Dense(4096, 'relu', dtype=data_type)(decoder)
    decoder = layers.Dense(vector_size, 'tanh', dtype=data_type)(decoder)

    decoder_model = Model(inputs=(decoder_vector_input, decoder_label_input), outputs=decoder, name='decoder')

    model = Autoencoder(encoder=encoder_model, decoder=decoder_model, latent_dimension=latent_dimension)
    if summary:
        print(encoder_model.summary())
        print(decoder_model.summary())
    return model


def make_cnn_autoencoder(
        vector_size, latent_dimension, summary=False, data_type='float32') -> Model:
    encoder = Sequential([
        layers.Reshape((vector_size, 1), input_shape=(vector_size,)),
        layers.Conv1D(32, 3, strides=2, padding='same', activation='relu'),
        layers.Conv1D(32, 3, strides=2, padding='same', activation='relu'),
        layers.Conv1D(1, 3, strides=2, padding='same', activation='relu')
    ])
    decoder = Sequential([
        layers.Conv1DTranspose(32, 3, strides=2, padding='same', activation='relu'),
        layers.Conv1DTranspose(32, 3, strides=2, padding='same', activation='relu'),
        layers.Conv1DTranspose(1, 3, strides=2, padding='same', activation='tanh', dtype=data_type),
        layers.Reshape((vector_size,))
    ])
    model = Autoencoder(encoder=encoder, decoder=decoder, latent_dimension=latent_dimension)
    if summary:
        print(encoder.summary())
        print(decoder.summary())
    return model


def make_fcc_variationalautoencoder(
        vector_size, latent_dimension, summary=False, data_type='float32') -> Model:
    encoder = Sequential([
        layers.Dense(128, 'relu', input_shape=(vector_size,), name='encoder_layer_1'),
        layers.Dense(64, 'relu', name='encoder_layer_2'),
        layers.Dense(latent_dimension * 2, name='encoder_layer_last', dtype=data_type),
    ], name='encoder')
    # encoder =
        # layers.Dense(256, 'relu', input_shape=(vector_size,)),
        # layers.Dense(256, 'relu', dtype=data_type),
        # layers.Dense(latent_dimension, 'tanh', dtype=data_type)

    decoder = Sequential([
        # layers.Dense(128, 'relu', input_shape=(latent_dimension,)),
        # layers.Dense(256, 'relu', dtype=data_type),
        # layers.Dense(vector_size, 'tanh', dtype=data_type)
        layers.Dense(64, 'relu', input_shape=(latent_dimension,), name='decoder_layer_1'),
        layers.Dense(128, 'relu', name='decoder_layer_2'),
        layers.Dense(vector_size, activation='tanh', name='decoder_layer_last', dtype=data_type)
    ], name='decoder')
    model = VAE(encoder=encoder, decoder=decoder, latent_dimension=latent_dimension)
    if summary:
        print(encoder.summary())
        print(decoder.summary())
    return model


def make_conditional_fcc_variationalautoencoder(
        vector_size, latent_dimension, num_frequencies, summary=False, data_type='float32') -> Model:
    encoder_vector_input = layers.Input((vector_size,))
    encoder_label_input = layers.Input((num_frequencies,))

    encoder_input = layers.Concatenate()([encoder_label_input, encoder_vector_input])

    encoder = layers.Dense(128, 'relu', input_shape=(vector_size,), name='encoder_layer_1')(encoder_input)
    encoder = layers.Dense(64, 'relu', name='encoder_layer_2')(encoder)
    encoder = layers.Dense(latent_dimension * 2, name='encoder_layer_last', dtype=data_type)(encoder)

    encoder_model = Model(inputs=(encoder_vector_input, encoder_label_input), outputs=encoder, name='encoder')

    decoder_vector_input = layers.Input((latent_dimension,))
    decoder_label_input = layers.Input((num_frequencies,))

    decoder_input = layers.Concatenate()([decoder_label_input, decoder_vector_input])

    decoder = layers.Dense(64, 'relu', input_shape=(latent_dimension,), name='decoder_layer_1')(decoder_input)
    decoder = layers.Dense(128, 'relu', name='decoder_layer_2')(decoder)
    decoder = layers.Dense(vector_size, activation='tanh', name='decoder_layer_last', dtype=data_type)(decoder)

    decoder_model = Model(inputs=(decoder_vector_input, decoder_label_input), outputs=decoder, name='decoder')

    model = VAE(encoder=encoder_model, decoder=decoder_model, latent_dimension=latent_dimension)
    if summary:
        print(encoder_model.summary())
        print(decoder_model.summary())
    return model



def make_cnn_variationalautoencoder(
        vector_size, latent_dimension, summary=False, data_type='float32') -> Model:
    encoder = Sequential([
        layers.Reshape((vector_size, 1), input_shape=(vector_size,)),
        layers.Conv1D(32, 3, strides=2, padding='same', activation='relu'),
        layers.Conv1D(32, 3, strides=2, padding='same', activation='relu'),
        layers.Conv1D(1, 3, strides=2, padding='same', activation='relu')
    ])
    decoder = Sequential([
        layers.Conv1DTranspose(32, 3, strides=2, padding='same', activation='relu'),
        layers.Conv1DTranspose(32, 3, strides=2, padding='same', activation='relu'),
        layers.Conv1DTranspose(1, 3, strides=2, padding='same', activation='tanh'),
        layers.Reshape((vector_size,))
    ])
    model = VAE(encoder=encoder, decoder=decoder, latent_dimension=latent_dimension)
    if summary:
        print(encoder.summary())
        print(decoder.summary())
    return model
