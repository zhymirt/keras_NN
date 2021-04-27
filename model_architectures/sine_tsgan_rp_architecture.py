import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Conv1D,
                                     Conv1DTranspose, Conv2D, Conv2DTranspose,
                                     Dense, Flatten, Input, LeakyReLU, Reshape)

image_shape, flattened_image_shape = (100, 100,), (int(1e4),)

def make_spectrogram_discriminator(data_type='float32', summary=False):
    model = Sequential([
        Reshape(image_shape+(1,), input_shape=image_shape),
        # 5 kernel or 2 3 kernels stacked
        # Conv2D(16, (1, 5), strides=(1, 2)),
        # LeakyReLU(alpha=0.2),
        Conv2D(8, 3, strides=1, padding='same'),
        Conv2D(8, 3, strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        # end of choice
        Conv2D(16, 3, strides=1, padding='valid'),
        LeakyReLU(alpha=0.2),
        Conv2D(16, 3, strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        Conv2D(64, 3, strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        Conv2D(64, 3, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        Conv2D(64, 3, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        Conv2D(64, 3, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        Conv2D(64, 3, strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        # Conv2D(64, 3, strides=1, padding='valid'),
        # LeakyReLU(alpha=0.2),
        Conv2D(1, 3),
        LeakyReLU(alpha=0.2),
        Flatten(),
        Dense(32),
        LeakyReLU(alpha=0.2),
        Dense(16),
        LeakyReLU(alpha=0.2),
        Dense(1, dtype=data_type)
    ], name='discriminator_1')
    if summary:
        print(model.input_shape)
        print(model.summary())
        print(model.output_shape)
    return model

def make_spectrogram_generator(latent_input, data_type='float32', summary=False):
    model = Sequential([
        Dense(64, input_shape=(latent_input,)),
        LeakyReLU(alpha=0.2),
        # layers.Dense(32),
        # layers.LeakyReLU(alpha=0.2),
        Dense(6**2),
        LeakyReLU(alpha=0.2),
        Reshape((6,)*2+(1,)),
        # layers.Conv2DTranspose(32, 3, strides=2),
        # layers.BatchNormalization(),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2DTranspose(32, 3, strides=2),
        # layers.BatchNormalization(),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2DTranspose(32, 3, strides=2, padding='same'),
        # layers.BatchNormalization(),
        # layers.LeakyReLU(alpha=0.2),
        Conv2DTranspose(32, 3, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        # Conv2DTranspose(32, 3, strides=2, padding='same'),
        # BatchNormalization(),
        # LeakyReLU(alpha=0.2),
        Conv2DTranspose(64, 3, strides=2),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv2DTranspose(128, 3, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv2DTranspose(1, 3, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        # Conv2DTranspose(32, 3, strides=2, padding='same'),
        # BatchNormalization(),
        # LeakyReLU(alpha=0.2),
        # Conv2DTranspose(1, 3, strides=2, padding='same'),
        # BatchNormalization(),
        # LeakyReLU(alpha=0.2),
        Reshape(image_shape, dtype=data_type),
    ], name='generator_1')
    if summary:
        print(model.input_shape)
        print(model.summary())
        print(model.output_shape)
    return model

def make_sine_wgan_discriminator(output_size, data_type='float32', summary=False):
    discriminator_2_spec_input = Input(shape=image_shape)
    # discriminator_2_spec = layers.Embedding(10, 50)(discriminator_2_spec_input)
    discriminator_2_spec = Reshape(image_shape+(1,))(discriminator_2_spec_input)
    discriminator_2_spec = Conv2D(8, 3, strides=1, padding='same')(discriminator_2_spec)
    discriminator_2_spec = Conv2D(8, 3, strides=2, padding='same')(discriminator_2_spec)
    discriminator_2_spec = LeakyReLU(alpha=0.2)(discriminator_2_spec)
    discriminator_2_spec = Conv2D(64, 3, strides=2, padding='valid')(discriminator_2_spec)
    discriminator_2_spec = LeakyReLU(alpha=0.2)(discriminator_2_spec)
    discriminator_2_spec = Conv2D(1, 3, strides=2, padding='same')(discriminator_2_spec)
    discriminator_2_spec = LeakyReLU(alpha=0.2)(discriminator_2_spec)
    discriminator_2_spec = Flatten()(discriminator_2_spec)
    discriminator_2_spec = Dense(output_size)(discriminator_2_spec)
    discriminator_2_spec = Reshape((output_size, 1))(discriminator_2_spec)

    discriminator_2_vector_input = Input(shape=(output_size,))
    discriminator_2_vector = Reshape((output_size, 1))(discriminator_2_vector_input)

    discriminator_2 = Concatenate()([discriminator_2_vector, discriminator_2_spec])
    discriminator_2 = Conv1D(8, 3, strides=1, padding='same')(discriminator_2)
    discriminator_2 = Conv1D(8, 3, strides=2, padding='same')(discriminator_2)
    discriminator_2 = LeakyReLU(alpha=0.2)(discriminator_2)
    discriminator_2 = Conv1D(64, 3, strides=2, padding='valid')(discriminator_2)
    discriminator_2 = LeakyReLU(alpha=0.2)(discriminator_2)
    discriminator_2 = Conv1D(1, 3, padding='same')(discriminator_2)
    discriminator_2 = LeakyReLU(alpha=0.2)(discriminator_2)
    discriminator_2 = Flatten()(discriminator_2)
    discriminator_2 = Dense(32)(discriminator_2)
    discriminator_2 = Dense(1, dtype=data_type)(discriminator_2)
    discriminator_2 = Model(inputs=(discriminator_2_vector_input, discriminator_2_spec_input), outputs=discriminator_2, name="discriminator_2")
    if summary:
        print(discriminator_2.input_shape)
        print(discriminator_2.summary())
        print(discriminator_2.output_shape)
    return discriminator_2

def make_sine_wgan_generator(latent_input, output_size, data_type='float32', summary=False):
    generator_2_spec_input = Input(shape=image_shape)
    generator_2_spec = Reshape(image_shape+(1,))(generator_2_spec_input)

    generator_2_vec_input = Input((latent_input,))
    # generator_2_vec = layers.Dense(flattened_image_shape[0])(generator_2_vec_input)
    # from generator 1
    generator_2_vec = Dense(64)(generator_2_vec_input)
    generator_2_vec = LeakyReLU(alpha=0.2)(generator_2_vec)
    generator_2_vec = Dense(6**2)(generator_2_vec)
    generator_2_vec = LeakyReLU(alpha=0.2)(generator_2_vec)
    generator_2_vec = Reshape((6,)*2+(1,))(generator_2_vec)
    generator_2_vec = Conv2DTranspose(32, 3, strides=2, padding='same')(generator_2_vec)
    generator_2_vec = BatchNormalization()(generator_2_vec)
    generator_2_vec = LeakyReLU(alpha=0.2)(generator_2_vec)
    generator_2_vec = Conv2DTranspose(64, 3, strides=2)(generator_2_vec)
    generator_2_vec = BatchNormalization()(generator_2_vec)
    generator_2_vec = LeakyReLU(alpha=0.2)(generator_2_vec)
    generator_2_vec = Conv2DTranspose(128, 3, strides=2, padding='same')(generator_2_vec)
    generator_2_vec = BatchNormalization()(generator_2_vec)
    generator_2_vec = LeakyReLU(alpha=0.2)(generator_2_vec)
    generator_2_vec = Conv2DTranspose(1, 3, strides=2, padding='same')(generator_2_vec)
    generator_2_vec = BatchNormalization()(generator_2_vec)
    generator_2_vec = LeakyReLU(alpha=0.2)(generator_2_vec)
    generator_2_vec = Reshape(image_shape, dtype=data_type)(generator_2_vec)
    # end of generator 1 architecture
    generator_2_vec = Reshape(image_shape+(1,))(generator_2_vec)

    generator_2 = Concatenate()([generator_2_vec, generator_2_spec])
    generator_2 = Conv2D(16, 3, strides=2)(generator_2)
    generator_2 = Conv2D(16, 3, strides=2)(generator_2)
    generator_2 = Conv2D(1, 3)(generator_2)
    generator_2 = Flatten()(generator_2)
    generator_2 = Dense(64)(generator_2)
    generator_2 = LeakyReLU(alpha=0.2)(generator_2)
    generator_2 = Reshape((64, 1))(generator_2)
    generator_2 = Conv1DTranspose(16, 3, strides=3)(generator_2)
    generator_2 = BatchNormalization()(generator_2)
    generator_2 = LeakyReLU(alpha=0.2)(generator_2)
    generator_2 = Conv1DTranspose(16, 3, strides=3)(generator_2)
    generator_2 = BatchNormalization()(generator_2)
    generator_2 = LeakyReLU(alpha=0.2)(generator_2)
    generator_2 = Conv1D(1, 3)(generator_2)
    generator_2 = Flatten()(generator_2)
    generator_2 = Dense(32, activation=tf.cos)(generator_2)
    generator_2 = Dense(output_size, activation='tanh', dtype=data_type)(generator_2)
    generator_2 = Model(inputs=(generator_2_vec_input, generator_2_spec_input), outputs=generator_2, name="generator_2")
    if summary:
        print(generator_2.input_shape)
        print(generator_2.summary())
        print(generator_2.output_shape)
    return generator_2
