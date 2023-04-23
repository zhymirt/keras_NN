import argparse
import os
import numpy as np
import scipy
import tensorflow as tf

from matplotlib import pyplot as plt
from pyts.image import RecurrencePlot
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from typing import List
from scipy.stats import wasserstein_distance

from AF_gan import plot_data
from utils.tensorflow_utils import data_to_dataset
from AF_gan import (normalize_data, denormalize_data, metric_fft_score,
                    get_fft_score, get_cross_correlate_score, plot_correlations, plot_recurrence_diff)
from constants.af_accel_constants import STANDARD_MODE, EBGAN_MODE, VAE_MODE, H_PARAMS, CONFIG_DATA, DTYPE, LATENT_DIM, \
    EPOCHS, DATA_DIR, DATA_PATH, TIME_PATH, LABEL_PATH, DATA_LENGTH, BATCH_SIZE
from scoring import tf_avg_wasserstein, average_wasserstein
from keras_data import get_date_string
from sine_gan import generate_sine
from custom_functions.custom_losses import ebgan_loss_fn
from custom_functions.custom_callbacks import PrintLogsCallback, FFTCallback
from custom_functions.custom_classes import WGAN, CWGAN, EBGAN, Hyperparameters, Data
from model_architectures.af_accel_GAN_architecture import (
    make_af_accel_discriminator, make_af_accel_generator,
    make_conditional_af_accel_discriminator,
    make_conditional_af_accel_generator, make_af_accel_fcc_generator,
    make_fcc_autoencoder, make_cnn_autoencoder,
    make_fcc_variationalautoencoder, make_conditional_fcc_autoencoder)
from utils.toml_utils import load_toml


def load_data(filename: str, separate_time: bool = True) -> np.ndarray:
    """ Return data loaded in from text file.

        :param filename: Filepath to load in
        :type filename: str
        :param bool separate_time: Whether to return time as separate array. Defaults to True
        :return: Return table of data, or tuple of time and data if separate_time is True
        :rtype: np.ndarray or tuple[np.ndarray, np.ndarray]
        """
    fn_data = np.loadtxt(filename, delimiter=',', skiprows=2)
    return (fn_data[:, 0], fn_data[:, 1:]) if separate_time else fn_data


def load_data_files(filenames: List[str], separate_time: bool = True) -> np.ndarray:
    """ Return array of data loaded in from text file.

    :param filenames: List of file paths to load data
    :type filenames: List[str]
    :param bool separate_time: Whether to return time as separate array. Defaults to True
    :return: Return table of data, or tuple of time and data if separate_time is True
    :rtype: np.ndarray or tuple[np.ndarray, np.ndarray]

    Parameters
    ----------
    filenames
    separate_time

    Returns
    -------

    """
    fn_data = np.array(
        [load_data(name, separate_time=False) for name in filenames])
    return (fn_data[:, :, 0], fn_data[:, :, 1:]) if separate_time else fn_data


def prepare_data(complete: np.ndarray, scaling: str = None, return_labels: bool = False) -> dict:
    """ Reorganize data to usable shape and return as parts in dictionary.

    :param np.ndarray complete: Complete array of time and data
    :param scaling: str or None. Defaults to None
    :param bool return_labels: Whether to include labels in dictionary. Defaults to False
    :return: Dictionary of data with keys: ['data', 'times', 'labels', 'normalized', 'scalars']
    :rtype: dict
    :raises AssertionError: Array has incompatible number of dimensions
    :raises NotImplementedError: scaling string given that is not implemented.
    Parameters
    ----------
    complete: np.ndarray - Array of data, including time column
    scaling: str - can only be None or 'normalize'
    return_labels: Include labels if True

    Returns
    -------
    dict: Contains keys ['data', 'times', 'labels', 'normalized', 'scalars']
    """
    # print(f'Number of dimensions: {complete.ndim}')
    # print(f'Shape: {complete.shape}')
    if not (complete.ndim == 2 or complete.ndim == 3):
        raise AssertionError(f'Array must be ndim 2 or 3, not {complete.ndim}')
    returned_values, full_data, labels = dict(), list(), list()
    data_start, data_end = 1, 5
    if complete.ndim == 2:
        full_time = complete[:, 0]
        for test_num, test in enumerate(
                complete.transpose((1, 0))[data_start:data_end], start=1):
            labels.append([test_num])
            full_data.append(test)
    elif complete.ndim == 3:
        full_time = complete[:, :, 0]
        for example_set in complete.transpose((0, 2, 1)):
            for test_num, test in enumerate(
                    example_set[data_start:data_end], start=1):
                labels.append([test_num])
                full_data.append(test)
    else:
        raise NotImplementedError
    full_data, labels = np.array(full_data), np.array(labels)
    returned_values['times']: np.ndarray = full_time  # complete[:, :, 0]
    returned_values['data']: np.ndarray = full_data
    if return_labels:
        returned_values['labels'] = labels
    if scaling is None:
        return returned_values
    elif scaling == 'normalize':
        returned_values['normalized'], returned_values['scalars'] = preprocessing.normalize(
            full_data, norm='max', axis=1, return_norm=True)  # normalize_data(full_data)
    # elif scaling == 'minmax':
    #     preprocessing.minmax_scale
    elif scaling == 'standardize':
        returned_values['normalized'] = preprocessing.scale(full_data, axis=1)
    else:
        raise NotImplementedError('Scaling version not allowed.')
    return returned_values


def plot_wasserstein_histogram(data: np.ndarray) -> plt.Figure:
    """ Plot histogram for wasserstein scores of data.

        Expects list of size two containing values.

        :param np.ndarray data: Data to be plotted. First dimension is [real, generated], second is data
        :return: Variable referencing matplotlib figure
        :rtype: plt.Figure
    """
    fig = plt.figure()
    # plt.hist(
    #     [np.squeeze(data[0]), np.squeeze(data[1])],
    #     label=['real data scores', 'synthetic data scores'])
    plt.subplot(1, 2, 1)
    plt.hist(np.squeeze(data[0]), label='real data scores')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(np.squeeze(data[1]), label='synthetic data scores')
    plt.legend()
    plt.tight_layout()
    return fig


# def frequency_wrapper(freq_func, fs):
#     """ Take three parameter frequency function and return two
#         parameter function.
#         """
#     pass


# TODO This function should be a class method
def save_gan(
        generator, discriminator, save_folder=None, generator_path=None,
        discriminator_path=None):
    """ Save GAN to folder.

        Paths can be empty, but None is not allowed right now

        :param generator: Generator model to save
        :type generator: tf.keras.Model
        :param discriminator: Discriminator model to be saved
        :type discriminator: tf.keras.Model
        :param save_folder: Root folder for discriminator and generator
        :type save_folder: None or str
        :param generator_path: Path to save generator
        :type generator_path: None or str
        :param discriminator_path: Path to save discriminator
        :type discriminator_path: None or str
    """
    folder = save_folder if save_folder is None else os.curdir
    generator.save(os.path.join(folder, generator_path))
    discriminator.save(os.path.join(folder, discriminator_path))


def plot_power_spectrum(data, fs):
    """ Plot power spectrum for multiple signals.

    :param data: Data to plot. Can have one or two dimensions
    :type data: np.ndarray
    :param fs: Sampling frequency for data
    :type fs: float
    :returns: None
    :rtype: None
    """
    # TODO instead of checking one or many signals, make singles expand themselves
    # Check that one or many signals
    # if one
    if np.ndim(data) == 1:
        plt.figure()
        frequencies, power = scipy.signal.periodogram(data, fs)
        plt.plot(frequencies, power)
    # if many
    elif np.ndim(data) == 2:
        for signal in data:
            plt.figure()
            frequencies, power = scipy.signal.periodogram(signal, fs)
            plt.plot(frequencies, power)
    plt.show()
    # plt.semilogy()


def plot_spectrogram(data, fs):
    """ Plot spectrograms for multiple signals.
    """
    # Check that one or many signals
    # if one
    if np.ndim(data) == 1:
        plt.figure()
        frequencies, times, spectrogram = scipy.signal.spectrogram(data, fs=fs, window=('tukey', 0.95))
        plt.pcolormesh(times, frequencies, spectrogram)
    # if many
    elif np.ndim(data) == 2:
        for signal in data:
            plt.figure()
            frequencies, times, spectrogram = scipy.signal.spectrogram(signal, fs)
            plt.pcolormesh(times, frequencies, spectrogram)
    plt.show()


def power_spectrum_score(dataset, synth, fs):
    pass


def standard_conditional(
        full_time, data, labels, data_size, data_type,
        latent_dimension, epochs, batch_size):
    """ Model with standard conditional architecture."""
    # Make labels
    data = data.astype(data_type)  # should be done ahead of time
    mlb = MultiLabelBinarizer()
    training_data = standard_conditional_data_prep(mlb, labels, data, data_type)
    # Make models
    num_tests = len(mlb.classes_)  # 4 This might be wrong, but we'll find out during testing, number of classes
    discriminator = make_conditional_af_accel_discriminator(
        data_size, num_tests)
    generator = make_conditional_af_accel_generator(
        latent_dimension, data_size, num_tests)
    cwgan = CWGAN(
        discriminator=discriminator, generator=generator,
        latent_dim=latent_dimension)
    cwgan.compile(  # todo learning rates should go in hyperparameters
        d_optimizer=keras.optimizers.Adam(learning_rate=0.001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        metrics=[metric_fft_score])
    cwgan.set_train_epochs(6, 1)  # 5
    # Pre train eval
    standard_conditional_pre_eval(generator, training_data[0:128], latent_dimension, mlb, data_type)
    # Train model
    early_stop = EarlyStopping(
        monitor='divergence_approx', mode='min', min_delta=1e-6, verbose=1,
        patience=5, restore_best_weights=True)
    cwgan.fit(training_data, epochs=epochs, batch_size=batch_size,
              callbacks=[FFTCallback(), early_stop], shuffle=True)
    # cwgan.fit(x=benign_data, y=labels, epochs=epochs, batch_size=batch_size, callbacks=callback_list)
    # model_dir = os.path.join(os.pardir, 'models', 'af_accel_sine_wave_models', '04_20_2023')
    # generator.save(os.path.join(model_dir, 'conditional_af_accel_generator_v11_alt_es'))
    # discriminator.save(os.path.join(model_dir, 'conditional_af_accel_discriminator_v11_alt_es'))
    # Eval
    rp = RecurrencePlot()
    eval_size = 64
    eval_labels = np.random.randint(0, 1, (eval_size, num_tests))
    standard_conditional_eval(
        generator, full_time, data, latent_dimension, eval_labels, mlb,
        eval_size, num_tests, data_type)


def standard_conditional_data_prep(mlb: MultiLabelBinarizer, labels, data, data_type=None, repeats=1):
    """ Data preparation for standard_conditional()"""
    new_labels = mlb.fit_transform(labels)
    print(f'Classes: {mlb.classes_}')
    print(f'labels shape: {labels.shape}, new labels shape: {new_labels.shape}')
    # Reused variables
    # repeat = repeats
    repeat = int(4e3)  # int(10e3)
    # Make more copies of reference data
    data_repeat = data.repeat(repeat, axis=0)  # 1e4
    new_labels = new_labels.repeat(repeat, axis=0)
    # Sprinkle in random noise
    data_repeat = data_repeat + np.random.normal(
        size=data_repeat.shape, scale=1e-10)
    # normalized = tf.convert_to_tensor(normalized)
    # new_labels = tf.convert_to_tensor(new_labels)
    # training_data = (data_repeat, new_labels)
    shuffled_data, shuffled_labels = shuffle(data_repeat, new_labels)
    dataset = (shuffled_data, shuffled_labels)
    return dataset


def standard_conditional_pre_eval(generator, data, latent_dimension, mlb, data_type=None):
    """ Pre-evaluation for standard conditional."""
    temp_label = mlb.transform([(1, 3)])
    generator.predict(
        (tf.zeros(shape=(1, latent_dimension)), tf.constant(temp_label)))
    print('FFT Score before training: {}'.format(
        get_fft_score(
            data[0:128],
            generator.predict((tf.random.normal(shape=(1, latent_dimension)),
                               tf.constant(mlb.transform([[1, 3]]), dtype=data_type))))))  # time,


def standard_conditional_train():
    """ Training steps for standard conditional."""
    pass


def standard_conditional_eval(
        generator, full_time, data, latent_dimension, labels, mlb, size, num_classes, data_type=None):
    """ Evaluation for standard conditional."""
    prediction = generator.predict((tf.random.normal(shape=(size, latent_dimension)), labels))
    # recurrence difference plot
    plot_recurrence_diff(data[labels[0]], prediction[0])
    # plt.show()
    plot_correlations(data[labels[0]], prediction[0])
    # plt.show()
    print(get_cross_correlate_score(data[0:size], prediction))
    print(get_fft_score(data[0:128], prediction[0:64]))  # time,
    for idx in range(num_classes):
        print('Current test: {}'.format(idx + 1))
        prediction = generator.predict(
            (tf.random.normal(shape=(4, latent_dimension)),
             tf.constant(mlb.transform([[idx + 1]]*4), dtype=data_type)))
        plot_data(
            full_time[idx], prediction,
            data[idx * num_classes: (idx * num_classes + 4)], show=False,
            save=False, save_path=os.path.join(results_dir, 'predictions.pdf'))
    prediction = generator.predict(
        (tf.random.normal(shape=(1, latent_dimension)),
         tf.constant(mlb.transform([[1, 3]]), dtype=data_type)))
    plot_data(full_time[0], prediction, data[0], show=False,
              save=False, save_path=os.path.join(results_dir, 'prediction.pdf'))

    plt.show()


def standard(
        full_time, data, data_size, data_type,
        latent_dimension, epochs, batch_size):
    """ Standard model for training."""
    # TODO make train and eval functions for each to further compartmentalize
    data_repeat = data.repeat(3e3, axis=0)  # 1e4
    print(f'Normalized shape: {data_repeat.shape}')
    dataset = data_to_dataset(data_repeat, dtype=data_type, batch_size=batch_size, shuffle=True)

    discriminator = make_af_accel_discriminator(data_size, data_type=data_type)
    # generator = make_af_accel_fcc_generator(latent_dimension, data_size, data_type=data_type)
    generator = make_af_accel_generator(latent_dimension, data_size, data_type=data_type)
    wgan = WGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dimension)
    wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                 g_optimizer=keras.optimizers.Adam(learning_rate=0.0004),
                 metrics=[metric_fft_score, tf_avg_wasserstein]
                 )
    wgan.set_train_epochs(4, 1)
    early_stop = EarlyStopping(monitor='metric_fft_score', mode='min', min_delta=1e-8, verbose=1, patience=5,
                               restore_best_weights=True)
    csv_log = CSVLogger(filename='./model_logs/{}_af_accel_log.csv'.format(get_date_string()))
    # checkpoint = keras.callbacks.ModelCheckpoint(filepath='./af_accel_tmp/checkpoint', save_weights_only=True)
    tb = keras.callbacks.TensorBoard(log_dir='./logs/af_accel_log_dir', histogram_freq=1)
    callback_list = [FFTCallback(), early_stop, PrintLogsCallback(), csv_log, tb]
    print('FFT Score before training: {}'.format(get_fft_score(data[0:128], generator.predict(
        tf.random.normal(shape=(batch_size, latent_dimension))))))  # time,
    print(average_wasserstein(data[0:2], generator.predict(
        tf.random.normal(shape=(2, latent_dimension)))[0:2]))
    plot_wasserstein_histogram([discriminator.predict(data[0:data.shape[0]]),
                                discriminator.predict(generator.predict(
                                    tf.random.normal(shape=(64, latent_dimension))))])
    wgan.fit(dataset, epochs=epochs, batch_size=batch_size, callbacks=callback_list)
    # Saving models
    # generator.save('models/af_accel_generator_full_new')
    # discriminator.save('models/af_accel_discriminator_full_new')
    # Eval
    plot_wasserstein_histogram([discriminator.predict(data[0:data.shape[0]]),
                                discriminator.predict(generator.predict(
                                    tf.random.normal(shape=(64, latent_dimension))))])
    plt.show()
    rp = RecurrencePlot()
    prediction = generator.predict(tf.random.normal(shape=(64, latent_dimension)))
    # recurrence difference plot
    plot_recurrence_diff(data[0], prediction[0])
    # plt.show()
    plot_correlations(data[0], prediction[0])
    # plt.show()
    print(get_cross_correlate_score(data[0], prediction[0]))
    print('FFT Score after training: {}'.format(
        get_fft_score(data[0:128], prediction[0:batch_size])))  # time,
    plot_data(full_time[0], prediction[0:4], data[0:4], show=True, save=False,
              save_path='./results/AF_5_23_21_')


def ebgan_conditional(
        full_time, data, labels, data_size, data_type,
        latent_dimension, epochs, batch_size):
    norm_repeat = data.repeat(4e3, axis=0)  # 1e4
    ae_early_stop = EarlyStopping(
        monitor='val_loss', mode='min', min_delta=1e-8, verbose=1,
        patience=5, restore_best_weights=True)
    eb_early_stop = EarlyStopping(
        monitor='Reconstruction error', mode='min', min_delta=1e-8,
        verbose=1, patience=7, restore_best_weights=True)
    autoencoder = make_conditional_fcc_autoencoder(data_size, latent_dimension)  # 16
    # dataset = data_to_dataset(data)
    autoencoder.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam')
    autoencoder.fit(norm_repeat, norm_repeat, epochs=256, batch_size=batch_size, validation_split=0.2,
                    callbacks=[ae_early_stop], shuffle=True)
    print("Training generator...")

    # Use autoencoder loss as gan loss function
    generator = make_af_accel_generator(
        latent_dimension, data_size, data_type=data_type)
    # generator = make_af_accel_fcc_generator(latent_dimension, data_size, data_type=data_type)
    ebgan_model = EBGAN(
        discriminator=autoencoder, generator=generator, latent_dim=latent_dimension)
    ebgan_model.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        d_loss_fn=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
        g_loss_fn=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
        metrics=[metric_fft_score, tf_avg_wasserstein, ebgan_model.d_loss_fn])
    print('Fitting model')
    ebgan_model.fit(
        norm_repeat, epochs=64, batch_size=batch_size, shuffle=True,
        callbacks=[eb_early_stop])


def ebgan(
        full_time, data, data_size, data_type,
        latent_dimension, epochs, batch_size):
    """ EBGAN model training."""
    norm_repeat = data.repeat(4e3, axis=0)  # 1e4
    ae_early_stop = EarlyStopping(
        monitor='val_loss', mode='min', min_delta=1e-8, verbose=1,
        patience=3, restore_best_weights=True)
    eb_early_stop = EarlyStopping(
        monitor='Reconstruction error', mode='min', min_delta=1e-12,
        verbose=1, patience=3, restore_best_weights=True)
    # Make autoencoder
    autoencoder = make_fcc_autoencoder(data_size, 16)
    dataset = data_to_dataset(data)
    autoencoder.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam')
    autoencoder.fit(norm_repeat, norm_repeat, epochs=256, batch_size=batch_size, validation_split=0.2,
                    callbacks=[ae_early_stop], shuffle=True)
    plot_data(full_time[0], autoencoder.predict(data)[0:6], data[0:6], show=True, save=False,
              save_path='./results/AF_5_23_21_')
    print("Training generator...")

    # Use autoencoder loss as gan loss function
    generator = make_af_accel_generator(
        latent_dimension, data_size, data_type=data_type)
    # generator = make_af_accel_fcc_generator(latent_dimension, data_size, data_type=data_type)
    ebgan_model = EBGAN(
        discriminator=autoencoder, generator=generator, latent_dim=latent_dimension)
    ebgan_model.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        d_loss_fn=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
        g_loss_fn=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
        metrics=[metric_fft_score, tf_avg_wasserstein, ebgan_model.d_loss_fn])
    print('Fitting model')
    ebgan_model.fit(
        norm_repeat, epochs=64, batch_size=batch_size, shuffle=True,
        callbacks=[eb_early_stop])
    # Saving models
    model_dir = os.path.join(os.pardir, 'models', '04_06_2023', 'ebgan')
    generator.save(os.path.join(model_dir, 'af_accel_generator_ebgan'))
    autoencoder.save(os.path.join(model_dir, 'af_accel_autoe_ebgan'))


def ebgan_eval(generator, full_time, data, latent_dimension, size, data_type=None):
    rp = RecurrencePlot()
    prediction = generator.predict(tf.random.normal(shape=(64, latent_dimension)))
    # recurrence difference plot
    plot_recurrence_diff(data[0], prediction[0])
    # plt.show()
    plot_correlations(data[0], prediction[0])
    # plt.show()
    print('Cross Correlation Score: {}'.format(
        get_cross_correlate_score(data[0], prediction[0])))
    print('FFT Score after training: {}'.format(
        get_fft_score(data[0:128], prediction[0:batch_size])))  # time,
    plot_data(
        full_time[0], prediction[0:8], generator.predict(prediction[0:8]),
        show=True, save=False, save_path='./results/AF_5_23_21_')


def vae(
        full_time, data, data_size, data_type,
        latent_dimension, epochs, batch_size):
    """ VAE training."""
    data_repeat = data.repeat(4e3, axis=0)  # 1e4
    ae_early_stop = EarlyStopping(
        monitor='loss', mode='min', min_delta=1e-8, verbose=1, patience=3,
        restore_best_weights=True)
    autoencoder = make_fcc_variationalautoencoder(data_size, 2)
    autoencoder.compile(optimizer='adam')
    autoencoder.fit(
        data_repeat, data_repeat, epochs=256, batch_size=batch_size,
        validation_split=0.2, callbacks=[ae_early_stop], shuffle=True)
    predictions = autoencoder.predict(data)
    plot_data(
        full_time[0], autoencoder.predict(data)[0:6], data[0:6], show=True,
        save=False, save_path='./results/AF_5_23_21_')


def run_model(
        mode, time, data, data_size, data_type, latent_dim, epochs, batch_size, labels=None):
    """ Train a model chosen by mode with given hyperparameters.

        :param str mode: Which model mode to run. Current options: [standard, ebgan, or vae]
        :param ndarray time:
        :param ndarray data:
        :param int data_size:
        :param data_type:
        :type data_type:
        :param int latent_dim:
        :param int epochs:
        :param int batch_size:
        :param labels:
        :type labels:
        :return: None
        """
    conditional = labels is not None
    if mode == 'standard':
        if conditional:
            standard_conditional(
                time, data, labels, data_size, data_type,
                latent_dim, epochs, batch_size)
        else:
            standard(
                time, data, data_size, data_type,
                latent_dim, epochs, batch_size)
    elif mode == 'ebgan':
        ebgan(
            time, data, data_size, data_type, latent_dim,
            epochs, batch_size)
    elif mode == 'vae':
        vae(
            time, data, data_size, data_type, latent_dim,
            epochs, batch_size)
    else:
        raise NotImplementedError(f'The mode {mode} is not a valid model option.')


def get_args():
    """ Parse command line."""
    # todo add ability to save models and results
    # todo config file will be default with command line overrides
    # todo separate getting args from main function
    parse = argparse.ArgumentParser()
    parse.add_argument('config_file')  # Path to config file
    parse.add_argument(
        '--mode', choices=[STANDARD_MODE, EBGAN_MODE, VAE_MODE], default=STANDARD_MODE)
    parse.add_argument('--conditional', action='store_true')  # Is model conditional
    parse.add_argument('--prepare_data', action='store_true')  # This calls prepare_data() which expects a specific file
    parse.add_argument('--make_paths', action='store_true')  # Make directories if they don't exist
    args = parse.parse_args()
    config_table = load_toml(args.config_file)
    config_data = config_table[CONFIG_DATA]
    config_hp = config_table[H_PARAMS]
    # todo these need to be made into constants
    conditional = config_table['training_mode']['conditional']
    mode = config_table['training_mode']['mode']
    # Data and time and labels will depend on version
    # Assume data, time, and labels are separate files
    # TODO check if files are numpy files, else do some work
    data_dir = config_data[DATA_DIR]
    data = np.load(os.path.join(data_dir, config_data[DATA_PATH]))
    time = np.load(os.path.join(data_dir, config_data[TIME_PATH]))
    if conditional:
        labels = np.load(os.path.join(data_dir, config_data[LABEL_PATH]))
        run_model(
            mode, time, data, DATA_LENGTH, config_data[DTYPE],
            config_hp[LATENT_DIM], config_hp[EPOCHS], config_hp[BATCH_SIZE],
            labels)
    else:
        run_model(
            mode, time, data, DATA_LENGTH, config_data[DTYPE],
            config_hp[LATENT_DIM], config_hp[EPOCHS], config_hp[BATCH_SIZE])


def get_hyperparameters(config_file):
    pass


def main():
    # Model parameters
    data_type, conditional, mode = 'float32', True, 'standard'
    latent_dimension, epochs, batch_size = 24, 64, 1
    # Load data
    folder_name = os.path.join(os.pardir, os.pardir, 'acceleration_data')
    file_names = ['accel_1.csv', 'accel_2.csv', 'accel_3.csv', 'accel_4.csv']
    complete_data = load_data_files(
        [os.path.join(folder_name, name) for name in file_names],
        separate_time=False)
    data_dict = prepare_data(complete_data, scaling='normalize', return_labels=True)
    # Making variables for loaded data
    full_time = data_dict['times']
    full_data, labels = np.array(data_dict['data']), np.array(data_dict['labels'])
    data_size = full_data.shape[1]
    normalized, scalars = data_dict['normalized'], data_dict['scalars']
    # Running chosen model
    run_model(
        mode, full_time, normalized, data_size, data_type,
        latent_dimension, epochs, batch_size,
        labels=(labels if conditional else None))


if __name__ == '__main__':
    get_args()
    exit()
    main()
