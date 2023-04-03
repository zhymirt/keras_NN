import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

image_shape, flattened_image_shape = (51, 1,), (51,)


def make_sine_tsgan_discriminator_1(summary=False, data_type='float32'):
    model = keras.Sequential([
        layers.Reshape(image_shape+(1,), input_shape=image_shape),
        # 5 kernel or 2 3 kernels stacked
        # layers.Conv2D(16, (1, 5), strides=(1, 2)),
        # layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, (3, 1), strides=(1, 1)),
        # layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, (3, 1), strides=(2, 1)),
        layers.LeakyReLU(alpha=0.2),
        # end of choice
        layers.Conv2D(64, (3, 1), strides=(1, 1)),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, (3, 1), strides=(1, 1)),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 1), strides=(1, 1)),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (3, 1)),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dense(64),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(32),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, dtype=data_type)
    ], name='discriminator_1')
    if summary:
        print(model.input_shape)
        print(model.summary())
        print(model.output_shape)
    return model


def make_sine_tsgan_generator_1(latent_dimension, summary=False, data_type='float32'):
    model = keras.Sequential([
        layers.Dense(32, input_shape=(latent_dimension,)),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(16),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(3),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((1, 3, 1,)),
        layers.Conv2DTranspose(32, (1, 3), strides=(1, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(32, (1, 3), strides=(1, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(32, (1, 3), strides=(1, 2)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(1, (1, 3), strides=(1, 2)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(1, 3),
        # layers.LeakyReLU(alpha=0.2),
        layers.Reshape(image_shape, dtype=data_type),
    ], name='generator_1')
    if summary:
        print(model.input_shape)
        print(model.summary())
        print(model.output_shape)
    return model


def make_sine_tsgan_discriminator_2(vector_size, summary=False):
    discriminator_2_spec_input = layers.Input(shape=image_shape)
    # discriminator_2_spec = layers.Embedding(10, 50)(discriminator_2_spec_input)
    discriminator_2_spec = layers.Reshape(image_shape+(1,))(discriminator_2_spec_input)
    discriminator_2_spec = layers.Conv2D(16, (3, 1), strides=(1, 1))(discriminator_2_spec)
    discriminator_2_spec = layers.Conv2D(16, (3, 1), strides=(2, 1))(discriminator_2_spec)
    discriminator_2_spec = layers.LeakyReLU(alpha=0.2)(discriminator_2_spec)
    # discriminator_2_spec = layers.Conv2D(16, (1, 3), strides=(1, 2))(discriminator_2_spec)
    # discriminator_2_spec = layers.LeakyReLU(alpha=0.2)(discriminator_2_spec)
    discriminator_2_spec = layers.Conv2D(1, (2, 1), strides=(2, 1))(discriminator_2_spec)
    discriminator_2_spec = layers.LeakyReLU(alpha=0.2)(discriminator_2_spec)
    discriminator_2_spec = layers.Flatten()(discriminator_2_spec)
    discriminator_2_spec = layers.Dense(vector_size)(discriminator_2_spec)
    discriminator_2_spec = layers.Reshape((vector_size, 1))(discriminator_2_spec)

    discriminator_2_vector_input = layers.Input(shape=(vector_size,))
    discriminator_2_vector = layers.Reshape((vector_size, 1))(discriminator_2_vector_input)

    discriminator_2 = layers.Concatenate()([discriminator_2_vector, discriminator_2_spec])
    discriminator_2 = layers.Conv1D(64, 3, strides=1)(discriminator_2)
    discriminator_2 = layers.Conv1D(64, 3, strides=2)(discriminator_2)
    discriminator_2 = layers.LeakyReLU(alpha=0.2)(discriminator_2)
    discriminator_2 = layers.Conv1D(64, 3, strides=2)(discriminator_2)
    discriminator_2 = layers.LeakyReLU(alpha=0.2)(discriminator_2)
    discriminator_2 = layers.Conv1D(1, 3)(discriminator_2)
    discriminator_2 = layers.LeakyReLU(alpha=0.2)(discriminator_2)
    discriminator_2 = layers.Conv1D(1, 3)(discriminator_2)
    discriminator_2 = layers.LeakyReLU(alpha=0.2)(discriminator_2)
    discriminator_2 = layers.Flatten()(discriminator_2)
    discriminator_2 = layers.Dense(32)(discriminator_2)
    discriminator_2 = layers.Dense(1)(discriminator_2)
    discriminator_2 = keras.Model(inputs=(discriminator_2_vector_input, discriminator_2_spec_input), outputs=discriminator_2, name="discriminator_2")
    # discriminator_2 = keras.Sequential([
    #     layers.Reshape((vector_size, 1,), input_shape=(vector_size,)),
    #     layers.Conv1D(64, 5, strides=3),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Conv1D(64, 3, strides=2),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Conv1D(1, 3),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Flatten(),
    #     layers.Dense(32),
    #     layers.Dense(1)
    # ], name='discriminator_2')
    if summary:
        print(discriminator_2.input_shape)
        print(discriminator_2.summary())
        print(discriminator_2.output_shape)
    return discriminator_2


def make_sine_tsgan_generator_2(latent_dimension, vector_size, summary=False, data_type='float32'):
    generator_2_spec_input = layers.Input(shape=image_shape)
    # generator_2_spec = layers.Embedding(10, 20)(generator_2_spec_input)
    # generator_2_spec = layers.Flatten()(generator_2_spec_input) # (generator_2_spec)
    # generator_2_spec = layers.Dense(flattened_image_shape[0])(generator_2_spec)
    generator_2_spec = layers.Reshape(image_shape+(1,))(generator_2_spec_input)

    generator_2_vec_input = layers.Input((latent_dimension,))
    generator_2_vec = layers.Dense(flattened_image_shape[0])(generator_2_vec_input)
    generator_2_vec = layers.Reshape(image_shape+(1,))(generator_2_vec)

    generator_2 = layers.Concatenate()([generator_2_vec, generator_2_spec])
    generator_2 = layers.Conv2D(16, (3, 1), strides=(2, 1))(generator_2)
    generator_2 = layers.Conv2D(16, (3, 1), strides=(2, 1))(generator_2)
    generator_2 = layers.Conv2D(1, (3, 1))(generator_2)
    generator_2 = layers.Flatten()(generator_2)
    generator_2 = layers.Dense(64)(generator_2)
    generator_2 = layers.LeakyReLU(alpha=0.2)(generator_2)
    generator_2 = layers.Reshape((64, 1))(generator_2)
    generator_2 = layers.Conv1DTranspose(16, 3, strides=3)(generator_2)
    generator_2 = layers.BatchNormalization()(generator_2)
    generator_2 = layers.LeakyReLU(alpha=0.2)(generator_2)
    generator_2 = layers.Conv1DTranspose(16, 3, strides=3)(generator_2)
    generator_2 = layers.BatchNormalization()(generator_2)
    generator_2 = layers.LeakyReLU(alpha=0.2)(generator_2)
    generator_2 = layers.Conv1D(1, 3)(generator_2)
    generator_2 = layers.Flatten()(generator_2)
    generator_2 = layers.Dense(32, activation=tf.cos)(generator_2)
    generator_2 = layers.Dense(vector_size, activation='tanh', dtype=data_type)(generator_2)
    generator_2 = keras.Model(inputs=(generator_2_vec_input, generator_2_spec_input), outputs=generator_2, name="generator_2")
    # generator_2 = keras.Sequential([
    #     layers.Reshape((1,)+image_shape, input_shape=image_shape),
    #     layers.Conv1D(16, 3, strides=2),
    #     layers.Conv1D(16, 3, strides=2),
    #     layers.Conv1D(1, 3),
    #     layers.Flatten(),
    #     layers.Dense(64),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Reshape((64, 1)),
    #     layers.Conv1DTranspose(16, 3, strides=3, dtype=data_type),
    #     layers.BatchNormalization(),
    #     layers.LeakyReLU(alpha=0.2, dtype=data_type),
    #     layers.Conv1DTranspose(16, 3, strides=3, dtype=data_type),
    #     layers.BatchNormalization(),
    #     layers.LeakyReLU(alpha=0.2, dtype=data_type),
    #     layers.Conv1D(1, 3, dtype=data_type),
    #     layers.Flatten(),
    #     layers.Dense(32, activation=tf.cos, dtype=data_type),
    #     layers.Dense(vector_size, activation='tanh', dtype=data_type)
    # ], name='generator_2')
    if summary:
        print(generator_2.input_shape)
        print(generator_2.summary())
        print(generator_2.output_shape)
    return generator_2


