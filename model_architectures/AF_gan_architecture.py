import tensorflow.keras as keras
from tensorflow import cos as cos
from tensorflow.keras import layers
from tensorflow.python.keras.engine.sequential import Sequential


def make_AF_discriminator(num_data_types, vector_size, summary=False):
    discriminator = keras.Sequential(
    [

        layers.Reshape((vector_size, num_data_types), input_shape=(vector_size, num_data_types,)),
        # layers.Conv2D(32, (3, 1), strides=(2, 1), padding='valid'),
        # layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(64, 5, strides=5, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(16, 5, strides=5, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(32, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(32, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(32, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(64, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(64, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(64, 3, 1, padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(64, 3, 1, padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(64, 3, 1, padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),

        # layers.Reshape((vector_size, num_data_types, 1), input_shape=(vector_size, num_data_types,)),
        # # layers.Conv2D(32, (3, 1), strides=(2, 1), padding='valid'),
        # # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(64, (5, 1), strides=(5, 1), padding='same'),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(16, (5, 1), strides=(5, 1), padding='same'),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(32, (3, 1), strides=(2, 1), padding='same'),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(32, (3, 1), strides=(2, 1), padding='same'),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(32, (3, 1), strides=(2, 1), padding='same'),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(64, (3, 1), strides=(2, 1), padding='same'),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(64, (3, 1), strides=(2, 1), padding='same'),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(64, (3, 1), (1, 1), padding='valid'),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(64, (3, 1), (1, 1), padding='valid'),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(64, (3, 1), (1, 1), padding='valid'),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(256, (1, 5), (1, 1), padding='valid'),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Flatten(),
        # layers.Dense(32),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Dense(16),
        # layers.LeakyReLU(alpha=0.2),
        layers.Dense(1)
    ],
    name="discriminator",
    )
    if summary:
        discriminator.summary()
    return discriminator


def make_AF_generator(latent_dimension, num_data_types, data_size, summary=False, data_type='float32'):
    mini_data, channels = 11, 16
    generator = keras.Sequential(
        [
            # keras.Input(shape=(latent_dimension,)),
            layers.Dense(mini_data * channels, input_shape=(latent_dimension,)),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((mini_data, channels)),
            # layers.Conv1DTranspose(64, 3, strides=2, padding='valid'),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(32, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(32, 5, strides=5, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(32, 5, strides=5, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(32, 3, strides=2, padding='valid'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(16, 5, strides=5, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1D(32, 3, strides=1, padding='valid', activation=cos),
            layers.BatchNormalization(),
            layers.Conv1D(5, 3, strides=1, padding='valid', activation='tanh'),
            layers.BatchNormalization(),
            layers.Reshape((data_size, num_data_types, 1)),
            layers.Conv2DTranspose(1, 1, strides=1, padding='valid', dtype=data_type),  # choose amplitude

            # layers.Dense(mini_data * 1 * channels, input_shape=(latent_dimension,)),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Reshape((mini_data, 1, channels)),
            # layers.Conv2DTranspose(16, (1, 5), strides=(1, 5), padding='same'),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(32, (3, 1), strides=(2, 1), padding='valid'),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(32, (3, 1), strides=(2, 1), padding='same'),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(16, (5, 1), strides=(5, 1), padding='same'),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(16, (5, 1), strides=(5, 1), padding='same'),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(32, (3, 1), strides=(2, 1), padding='valid'),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(16, (5, 1), strides=(5, 1), padding='same'),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2D(16, (3, 1), strides=(1, 1), padding='valid', activation=cos),
            # layers.BatchNormalization(),
            # layers.Conv2D(1, (3, 1), strides=(1, 1), padding='valid', activation='tanh'),
            # layers.BatchNormalization(),
            # layers.Conv2DTranspose(1, 1, strides=1, padding='same', dtype=data_type),  # choose amplitude

            # layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same'),
            # layers.Conv2DTranspose(8, (3, 1), strides=(2, 1)),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(8, (3, 1), strides=(2, 1), padding='same'),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(16, (3, 1), strides=(2, 1)),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(16, (3, 1), strides=(2, 1)),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(16, (3, 1), strides=(2, 1)),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(16, (3, 1), strides=(2, 1)),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(16, (3, 1), strides=(2, 1)),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(16, (3, 1), strides=(2, 1)),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(16, (3, 1), strides=(2, 1)),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2D(1, (3, 1)),
            layers.Reshape((data_size, num_data_types,))
        ],
        name="generator",
    )
    if summary:
        generator.summary()
    return generator


def make_AF_single_discriminator(vector_size, summary=False):
    model = Sequential([
        layers.Reshape((vector_size, 1), input_shape=(vector_size,)),
        layers.Conv1D(16, 3, strides=1, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(16, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(16, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(16, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(16, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(16, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(32, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(32, 3, strides=2, padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(64, 3, strides=2, padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(64, 3, strides=2, padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(1, 3, 1, padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dense(32),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(16),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1),
    ])
    if summary:
        model.summary()
    return model


def make_AF_single_generator(latent_dimension, data_size, summary=False):
    mini_data = 10
    generator = keras.Sequential(
        [
            layers.Conv1DTranspose(32, 3, strides=2, padding='same'),
            layers.Conv1DTranspose(32, 3, strides=2, padding='valid'),
            # layers.Dense(mini_data, input_shape=(latent_dimension,)),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Reshape((mini_data, 1,)),
            # layers.Conv1DTranspose(16, 3, strides=2),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv1DTranspose(16, 3, strides=2, padding='same'),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv1DTranspose(16, 3, strides=2),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv1DTranspose(16, 3, strides=2),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv1DTranspose(16, 3, strides=2),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv1DTranspose(16, 3, strides=2),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv1DTranspose(16, 3, strides=2),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv1DTranspose(64, 3, strides=2),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv1DTranspose(64, 3, strides=2),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv1D(1, 3),
            # layers.Reshape((data_size,))
        ],
        name="generator",
    )
    if summary:
        generator.summary()
    return generator


def make_AF_rp_discriminator(vector_size, summary=True):
    model = Sequential([
        layers.Reshape((vector_size,vector_size, 1), input_shape=(vector_size,vector_size,)),
        layers.Conv2D(16, 3, strides=1, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(16, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(16, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(16, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(16, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(16, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(32, 3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(32, 3, strides=2, padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, 3, strides=2, padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, 3, strides=2, padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, 3, 1, padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dense(32),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(16),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1),
    ])
    if summary:
        model.summary()
    return model


def make_AF_rp_generator(latent_input, image_shape, data_type='float32', summary=False):
    model = Sequential([
        layers.Dense(64, input_shape=(latent_input,)),
        layers.LeakyReLU(alpha=0.2),
        # layers.Dense(32),
        # layers.LeakyReLU(alpha=0.2),
        layers.Dense(5**2),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((5,)*2+(1,)),
        layers.Conv2DTranspose(16, 3, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(16, 3, strides=2, padding='valid'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(16, 3, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(16, 3, strides=2, padding='valid'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(32, 3, strides=2, padding='valid'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(32, 3, strides=2, padding='valid'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64, 3, strides=2, padding='valid'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64, 3, strides=2, padding='valid'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, 3, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(1, 3, strides=2, padding='valid'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape(image_shape, dtype=data_type)
    ], name='generator_1')
    if summary:
        model.summary()
    return model


def make_conditional_discriminator(summary=False):
    pass


def make_conditional_generator(summary=False):
    pass


image_shape, flattened_image_shape = (129, 24,), (3096,)
def make_AF_spectrogram_discriminator_1(data_type='float32', summary=False):
    model = keras.Sequential([
        layers.Reshape(image_shape+(1,), input_shape=image_shape),
        # 5 kernel or 2 3 kernels stacked
        # layers.Conv2D(16, (1, 5), strides=(1, 2)),
        # layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same'),
        # layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        # end of choice
        layers.Conv2D(64, (1, 3), strides=(1, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, (3, 1), strides=(1, 1), padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, (3, 1), strides=(1, 1), padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, (3, 1), strides=(1, 1), padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, (3, 1), strides=(1, 1), padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, (3, 1), strides=(1, 1), padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (3, 3)),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dense(32),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(16),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, dtype=data_type)
    ], name='discriminator_1')
    if summary:
        model.summary()
    return model


def make_AF_spectrogram_generator_1(latent_dimension, data_type='float32', summary=False):
    mini_data, channels = (3, 8), 16
    flattened = mini_data[0] * mini_data[1] * channels
    model = keras.Sequential([
        layers.Dense(flattened, input_shape=(latent_dimension,)),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Dense(32),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Dense(24),
        # layers.LeakyReLU(alpha=0.2),
        layers.Reshape((3, 8, channels)),
        layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation=cos),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(1, (1, 3), strides=(1, 2), activation='tanh'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(1, (1, 3)),
        # layers.LeakyReLU(alpha=0.2),
        layers.Reshape(image_shape, dtype=data_type),
    ], name='generator_1')
    if summary:
        model.summary()
    return model


def make_AF_spectrogram_discriminator_2(data_type='float32', summary=False):
    pass


def make_AF_spectrogram_generator_2(data_type='float32', summary=False):
    pass
