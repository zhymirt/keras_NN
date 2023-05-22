import collections

import numpy as np
import pytest
from tensorflow import keras

from model_architectures import af_accel_GAN_architecture
from custom_functions.custom_classes import CEBGAN, CWGAN, WGAN

Model_Params = collections.namedtuple(
    'Params', ['latent_dim', 'size', 'type'], )
Conditional_Model_Params = collections.namedtuple(
    'Params', ['latent_dim', 'size', 'num_classes', 'type'], )


# Parameters
@pytest.fixture
def get_params():
    latent_dim, data_size, dtype = 24, 5_000, 'float32'
    params = Model_Params(latent_dim, data_size, dtype)
    return params


@pytest.fixture
def get_conditional_params(get_params):
    # latent_dim, data_size, dtype = 24, 5_000, 'float32'
    # params = Model_Params(latent_dim, data_size, dtype)
    num_classes = 4
    params = Conditional_Model_Params(
        get_params.latent_dim, get_params.size,
        num_classes, get_params.type)
    return params


@pytest.fixture
def gcp(get_conditional_params):
    yield get_conditional_params


@pytest.fixture
def train_data(gcp):
    num_signals = 10
    data = np.random.normal(size=(num_signals, gcp.size))
    return data


@pytest.fixture
def labels(gcp):
    num_signals = 10
    labels = np.random.randint(0, 2, size=(num_signals, gcp.num_classes))
    return labels


@pytest.fixture
def conditional_training_data(train_data, labels):
    return train_data, labels


# Models
@pytest.fixture
def get_cwgan(gcp):
    discriminator = af_accel_GAN_architecture.make_conditional_af_accel_discriminator(
        gcp.size, gcp.num_classes)
    generator = af_accel_GAN_architecture.make_conditional_af_accel_generator(
        gcp.latent_dim, gcp.size, gcp.num_classes)
    cwgan = CWGAN(
        discriminator=discriminator, generator=generator,
        latent_dim=gcp.latent_dim)
    return cwgan


@pytest.fixture
def get_wgan(get_params):
    discriminator = af_accel_GAN_architecture.make_af_accel_discriminator(
        get_params.size, data_type=get_params.type)
    # generator = make_af_accel_fcc_generator(latent_dimension, data_size, data_type=data_type)
    generator = af_accel_GAN_architecture.make_af_accel_generator(
        get_params.latent_dim, get_params.size, data_type=get_params.type)
    wgan = WGAN(discriminator=discriminator, generator=generator, latent_dim=get_params.latent_dim)
    return wgan


@pytest.fixture
def get_cebgan(gcp):
    autoencoder = af_accel_GAN_architecture.make_conditional_fcc_autoencoder(
        gcp.size, gcp.latent_dim, gcp.num_classes)  # 16
    generator = af_accel_GAN_architecture.make_conditional_af_accel_generator(
        gcp.latent_dim, gcp.size, gcp.num_classes, data_type=gcp.type)
    return CEBGAN(
        discriminator=autoencoder, generator=generator,
        latent_dim=gcp.latent_dim)


@pytest.fixture
def get_cvae(gcp):
    autoencoder = af_accel_GAN_architecture.make_conditional_fcc_variationalautoencoder(
        gcp.size, gcp.latent_dim, gcp.num_classes)
    return autoencoder


@pytest.fixture
def get_vae(get_params):
    autoencoder = af_accel_GAN_architecture.make_fcc_variationalautoencoder(
        get_params.size, get_params.latent_dim)
    return autoencoder


# Tests
def test_cwgan_valid(get_cwgan):
    assert get_cwgan is not None


def test_cwgan_trains(get_cwgan, conditional_training_data, train_data, labels):
    epochs, batch_size = 1, 64
    get_cwgan.compile(  # todo learning rates should go in hyperparameters
        d_optimizer=keras.optimizers.Adam(learning_rate=0.001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002))
    get_cwgan.set_train_epochs(6, 1)  # 5
    history = get_cwgan.fit(conditional_training_data, epochs=epochs, batch_size=batch_size,
                            shuffle=True)


def test_wgan_valid(get_wgan):
    assert get_wgan is not None


def test_cebgan_valid(get_cebgan):
    assert get_cebgan is not None


def test_cebgan_autoencoder_trains(get_cebgan, conditional_training_data):
    epochs, batch_size = 2, 5
    get_cebgan.discriminator.compile(
        loss=keras.losses.mean_squared_error, optimizer='adam')
    get_cebgan.discriminator.fit(
        conditional_training_data, conditional_training_data, epochs=epochs,
        batch_size=batch_size, validation_split=0.2, shuffle=True)


def test_cebgan_trains(get_cebgan, conditional_training_data):
    epochs, batch_size = 2, 5
    get_cebgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        d_loss_fn=keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM),
        g_loss_fn=keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM))
    print('Fitting model')
    get_cebgan.fit(
        conditional_training_data, epochs=epochs, batch_size=batch_size, shuffle=True)


def test_vae_valid(get_vae):
    assert get_vae is not None


def test_vae_trains(get_vae, train_data):
    epochs, batch_size = 2, 5
    get_vae.compile(optimizer='adam')
    get_vae.fit(
        train_data, train_data, epochs=epochs, batch_size=batch_size,
        validation_split=0.2, shuffle=True)


def test_cvae_valid(get_cvae):
    assert get_cvae is not None


def test_cvae_trains(get_cvae, conditional_training_data):
    epochs, batch_size = 2, 5
    get_cvae.compile(optimizer='adam')
    get_cvae.fit(
        conditional_training_data, conditional_training_data, epochs=epochs, batch_size=batch_size,
        validation_split=0.2, shuffle=True)


if __name__ == '__main__':
    pass
