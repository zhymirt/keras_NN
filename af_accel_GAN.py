import os
import numpy as np
import scipy
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from matplotlib import pyplot as plt
from pyts.image import RecurrencePlot
from sklearn import preprocessing
from tensorflow import keras
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List
from scipy.stats import wasserstein_distance
from AF_gan import (normalize_data, denormalize_data, metric_fft_score,
                    get_fft_score, get_cross_correlate_score, plot_data,
                    plot_correlations, plot_recurrence_diff)
from custom_losses import ebgan_loss_fn
from keras_data import data_to_dataset, get_date_string
from custom_callbacks import PrintLogsCallback, FFTCallback
from custom_classes import WGAN, CWGAN, EBGAN
from model_architectures.af_accel_GAN_architecture import (
    make_af_accel_discriminator, make_af_accel_generator,
    make_conditional_af_accel_discriminator,
    make_conditional_af_accel_generator, make_af_accel_fcc_generator,
    make_fcc_autoencoder, make_cnn_autoencoder,
    make_fcc_variationalautoencoder)


def load_data(filename: str, separate_time: bool = True) -> np.ndarray:
    """ Return data loaded in from text file.

        :param filename: str
        :param separate_time: bool = True
        :return: np.ndarray"""
    fn_data = np.loadtxt(filename, delimiter=',', skiprows=2)

    return (fn_data[:, 0], fn_data[:, 1:]) if separate_time else fn_data
    # if separate_time:
    #     return fn_data[:, 0], fn_data[:, 1:]
    # else:
    #     return fn_data


def load_data_files(filenames: List[str], separate_time: bool = True) -> np.ndarray:
    """ Return array of data loaded in from text file.


    Parameters
    ----------
    filenames
    separate_time

    Returns
    -------

    """
    fn_data = np.array([load_data(name, separate_time=False) for name in filenames])

    return (fn_data[:, :, 0], fn_data[:, :, 1:]) if separate_time else fn_data
    # if separate_time:
    #     return fn_data[:, :, 0], fn_data[:, :, 1:]
    # else:
    #     return fn_data


def prepare_data(complete: np.ndarray, scaling: str = None, return_labels: bool = False) -> dict:
    """ Reorganize data to usable shape and return as parts in dictionary.

    Parameters
    ----------
    complete: np.ndarray - Array of data, including time column
    scaling: str - can only be None or 'normalize'
    return_labels: Include labels if True

    Returns
    -------
    dict: Contains keys ['data', 'times', 'labels', 'normalized', 'scalars']
    """
    returned_values, full_data, labels = dict(), list(), list()
    # print('Complete shape: {}'.format(complete.shape))
    full_time = complete[:, :, 0]
    data_start, data_end = 1, 5
    # print('Full time shape: {}'.format(full_time.shape))
    if complete.ndim == 2:
        for test_num, test in enumerate(
                complete.transpose((1, 0))[data_start:data_end], start=1):
            # if np.sum(np.square(test)) > 1e-8: # numbers aren't all zero
            # print('Test #{} shape: {}'.format(test_num + 1, test.shape))
            labels.append([test_num])
            full_data.append(test)
    elif complete.ndim == 3:
        for example_set in complete.transpose((0, 2, 1)):
            for test_num, test in enumerate(
                    example_set[data_start:data_end], start=1):
                # if np.sum(np.square(test)) > 1e-8: # numbers aren't all zero
                # print('Test #{} shape: {}'.format(test_num + 1, test.shape))
                labels.append([test_num])
                full_data.append(test)
    else:
        print("Cannot complete")
        return dict()
    full_data, labels = np.array(full_data), np.array(labels)
    returned_values['times'] = full_time
    returned_values['data'] = full_data
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
    return returned_values


def average_wasserstein(arr_1: np.ndarray, arr_2: np.ndarray) -> float:
    """ Calculate average wasserstein distance between two arrays."""
    arr_1, arr_2 = np.asarray(arr_1), np.asarray(arr_2)
    # print('Array shapes: {}, {}'.format(arr_1.shape, arr_2.shape))
    if arr_1.ndim == 1:
        return wasserstein_distance(arr_1, arr_2)
    elif arr_1.ndim == 2:
        distances = [wasserstein_distance(item_1, item_2)
                     for item_1, item_2 in zip(arr_1, arr_2)]
        # distances = list()
        # for item_1, item_2 in zip(arr_1, arr_2):
        #     distances.append(scipy.stats.wasserstein_distance(item_1, item_2))
        # print(distances)
        return np.asarray(distances).mean()


def tf_avg_wasserstein(arr_1: np.ndarray, arr_2: np.ndarray) -> tf.float32:
    """ Wrap average_wasserstein function and return result."""
    return tf.py_function(average_wasserstein, (arr_1, arr_2), tf.float32)


def plot_wasserstein_histogram(data: np.ndarray) -> plt.Figure:
    """ Plot histogram for wasserstein scores of data.
        Expects list of size two containing values."""
    fig = plt.figure()
    plt.hist(
        [np.squeeze(data[0]), np.squeeze(data[1])],
        label=['real data scores', 'synthetic data scores'])
    plt.legend()
    plt.tight_layout()
    return fig


def standard_conditional(
        full_time, data, labels, data_size, data_type,
        latent_dimension, epochs, batch_size):
    """ Model with standard conditional architecture."""
    # Preparation
    mlb = MultiLabelBinarizer()
    new_labels = mlb.fit_transform(labels)
    print('Classes: {}'.format(mlb.classes_))
    print('labels shape: {}, new labels shape: {}'.format(
        labels.shape, new_labels.shape))
    data = data.astype(data_type)
    repeat = int(4e3)
    data_repeat = data.repeat(repeat, axis=0)  # 1e4
    new_labels = new_labels.repeat(repeat, axis=0)
    data_repeat = data_repeat + tf.random.normal(
        shape=data_repeat.shape, stddev=1e-10, dtype=data_type)
    num_tests = 4
    # normalized = tf.convert_to_tensor(normalized)
    # new_labels = tf.convert_to_tensor(new_labels)
    discriminator = make_conditional_af_accel_discriminator(
        data_size, num_tests)
    generator = make_conditional_af_accel_generator(
        latent_dimension, data_size, num_tests)
    cwgan = CWGAN(
        discriminator=discriminator, generator=generator,
        latent_dim=latent_dimension)
    cwgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
                  metrics=[metric_fft_score, 'accuracy']
                  )
    cwgan.set_train_epochs(5, 1)
    # Pre train eval
    temp_label = mlb.transform([(1, 3)])
    generator.predict(
        (tf.zeros(shape=(1, latent_dimension)), tf.constant(temp_label)))
    early_stop = EarlyStopping(
        monitor='wasserstein_score', mode='min', min_delta=1e-6, verbose=1,
        patience=5, restore_best_weights=True)
    print('FFT Score before training: {}'.format(
        get_fft_score(
            data_repeat[0:128],
            generator.predict((tf.random.normal(shape=(1, latent_dimension)),
                               tf.constant(mlb.transform([[1, 3]]), dtype=data_type))))))  # time,
    # Train model
    cwgan.fit((data_repeat, new_labels), epochs=epochs, batch_size=batch_size,
              callbacks=[FFTCallback(), early_stop], shuffle=True)
    # cwgan.fit(x=benign_data, y=labels, epochs=epochs, batch_size=batch_size, callbacks=callback_list)
    # generator.save('./models/conditional_af_accel_generator_v3')
    # discriminator.save('./models/conditional_af_accel_discriminator_v3')
    # Eval
    rp = RecurrencePlot()
    eval_size = 64
    eval_labels = np.random.randint(0, 1, (eval_size, num_tests))
    prediction = generator.predict((tf.random.normal(shape=(eval_size, latent_dimension)), eval_labels))
    # recurrence difference plot
    plot_recurrence_diff(data[eval_labels[0]], prediction[0])
    # plt.show()
    plot_correlations(data[eval_labels[0]], prediction[0])
    # plt.show()
    print(get_cross_correlate_score(data[0:eval_size], prediction))
    print(get_fft_score(data[0:128], prediction[0:64]))  # time,
    for idx in range(num_tests):
        print('Current test: {}'.format(idx + 1))
        prediction = generator.predict(
            (tf.random.normal(shape=(4, latent_dimension)),
             tf.constant(mlb.transform([[idx + 1]]*4), dtype=data_type)))
        plot_data(
            full_time[idx], prediction,
            data[idx * num_tests: (idx * num_tests + 4)], show=False,
            save=False, save_path='./results/AF_5_23_21_')
    prediction = generator.predict(
        (tf.random.normal(shape=(1, latent_dimension)),
         tf.constant(mlb.transform([[1, 3]]), dtype=data_type)))
    plot_data(full_time[0], prediction, data[0], show=False,
              save=False, save_path='./results/AF_5_23_21_')

    plt.show()


def standard(
        full_time, data, data_size, data_type,
        latent_dimension, epochs, batch_size):
    """ Standard model for training."""
    data_repeat = data.repeat(3e3, axis=0)  # 1e4
    print('Normalized shape: {}'.format(data_repeat.shape))
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
    plot_wasserstein_histogram([discriminator.predict(data[0:data.shape[0]]),
                                discriminator.predict(generator.predict(
                                    tf.random.normal(shape=(64, latent_dimension))))])
    plt.show()
    # Saving models
    # generator.save('models/af_accel_generator_full_new')
    # discriminator.save('models/af_accel_discriminator_full_new')
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
        d_loss_fn=tf.keras.losses.mean_squared_error,
        g_loss_fn=tf.keras.losses.mean_squared_error,
        metrics=[metric_fft_score, tf_avg_wasserstein, 'g_loss'])
    ebgan_model.fit(
        norm_repeat, epochs=64, batch_size=batch_size, shuffle=True,
        callbacks=[eb_early_stop])  #
    # Saving models
    # generator.save('models/af_accel_generator_full')
    # discriminator.save('models/af_accel_discriminator_full')
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
        full_time[0], prediction[0:8], autoencoder.predict(prediction[0:8]),
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


def main():
    # Load data
    data_type, conditional, mode = 'float32', True, 'standard'
    latent_dimension, epochs, batch_size = 10, 64, 32
    folder_name = '../acceleration_data'
    file_names = ['accel_1.csv', 'accel_2.csv', 'accel_3.csv', 'accel_4.csv']
    complete_data = load_data_files(
        [os.path.join(folder_name, name) for name in file_names],
        separate_time=False)
    full_time = complete_data[:, :, 0]
    data_dict = prepare_data(complete_data, scaling='normalize', return_labels=True)
    # Making variables for loaded data
    full_data, labels = np.array(data_dict['data']), np.array(data_dict['labels'])
    # exit()
    data_size = full_data.shape[1]
    normalized, scalars = data_dict['normalized'], data_dict['scalars']
    # Running chosen model
    if mode == 'standard':
        if conditional:
            standard_conditional(
                full_time, normalized, labels, data_size, data_type,
                latent_dimension, epochs, batch_size)
        else:
            standard(
                full_time, normalized, data_size, data_type,
                latent_dimension, epochs, batch_size)
    if mode == 'ebgan':
        ebgan(
            full_time, normalized, data_size, data_type, latent_dimension,
            epochs, batch_size)
    if mode == 'vae':
        vae(full_time, normalized, data_size, data_type, latent_dimension, epochs, batch_size)


if __name__ == '__main__':
    main()
