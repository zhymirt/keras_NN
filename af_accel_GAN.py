import os
import numpy as np
import scipy.stats
import tensorflow as tf

from keras.callbacks import EarlyStopping, CSVLogger
from matplotlib import pyplot as plt
from pyts.image import RecurrencePlot
from tensorflow import keras
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List

from AF_gan import normalize_data, denormalize_data, metric_fft_score, get_fft_score, get_cross_correlate_score, \
    plot_data, plot_correlations, plot_recurrence_diff
from custom_losses import ebgan_loss_fn
from keras_data import data_to_dataset, get_date_string
from keras_gan import WGAN, fft_callback, cWGAN, print_logs_callback, EBGAN
from model_architectures.af_accel_GAN_architecture import make_af_accel_discriminator, make_af_accel_generator, \
    make_conditional_af_accel_discriminator, make_conditional_af_accel_generator, make_af_accel_fcc_generator, \
    make_fcc_autoencoder, make_cnn_autoencoder


def load_data(filename: str, separate_time: bool = True) -> np.ndarray:
    """ Return data loaded in from text file.

        :param filename: str
        :param separate_time: bool = True
        :return: np.ndarray"""
    fn_data = np.loadtxt(filename, delimiter=',', skiprows=2)
    if separate_time:
        return fn_data[:, 0], fn_data[:, 1:]
    else:
        return fn_data


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
    if separate_time:
        return fn_data[:, :, 0], fn_data[:, :, 1:]
    else:
        return fn_data


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
    returned_values = dict()
    print('Complete shape: {}'.format(complete.shape))
    full_time = complete[:, :, 0]
    full_data, labels = [], []
    print('Full time shape: {}'.format(full_time.shape))
    for example_set in complete.transpose((0, 2, 1)):
        for test_num, test in enumerate(example_set[1:5]):
            # if np.sum(np.square(test)) > 1e-8: # numbers aren't all zero
            # print('Test #{} shape: {}'.format(test_num + 1, test.shape))
            labels.append([test_num + 1])
            full_data.append(test)
    full_data, labels = np.array(full_data), np.array(labels)
    returned_values['times'] = full_time
    returned_values['data'] = full_data
    if return_labels:
        returned_values['labels'] = labels
    if scaling is not None and scaling == 'normalize':
        returned_values['normalized'], returned_values['scalars'] = normalize_data(full_data)
    return returned_values


def average_wasserstein(arr_1, arr_2):
    """" Calculate average wasserstein distance between two arrays."""
    arr_1, arr_2 = np.asarray(arr_1), np.asarray(arr_2)
    # print('Array shapes: {}, {}'.format(arr_1.shape, arr_2.shape))
    if arr_1.ndim == 1:
        return scipy.stats.wasserstein_distance(arr_1, arr_2)
    elif arr_1.ndim == 2:
        distances = list()
        for item_1, item_2 in zip(arr_1, arr_2):
            distances.append(scipy.stats.wasserstein_distance(item_1, item_2))
        # print(distances)
        return np.asarray(distances).mean()


def tf_avg_wasserstein(arr_1, arr_2):
    """ Wrap average_wasserstein function and return result."""
    return tf.py_function(average_wasserstein, (arr_1, arr_2), tf.float32)


def plot_wasserstein_histogram(data):
    """ Plot histogram for wasserstein scores of data.
        Expects list of size two containing values."""
    fig = plt.figure()
    plt.hist([np.squeeze(data[0]), np.squeeze(data[1])], label=['real data scores', 'synthetic data scores'])
    plt.legend()
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    latent_dimension, data_type, epochs, batch_size, conditional, mode = 128, 'float32', 64, 32, True, 'standard'
    # time, data = load_data('../acceleration_data/accel_1.csv')
    # print('Time shape: {}, Data shape: {}'.format(time.shape, data.shape))
    complete_data = load_data_files([os.path.join('../acceleration_data', name) for name in ('accel_1.csv',
                                                                                             'accel_2.csv',
                                                                                             'accel_3.csv',
                                                                                             'accel_4.csv')],
                                    separate_time=False)
    print('Complete shape: {}'.format(complete_data.shape))
    full_time = complete_data[:, :, 0]
    full_data, labels = [], []
    print('Full time shape: {}'.format(full_time.shape))
    data_dict = prepare_data(complete_data, scaling='normal', return_labels=True)
    full_data, labels = np.array(data_dict['data']), np.array(data_dict['labels'])
    # exit()
    # print('Complete shape: {}'.format(complete_data.shape))
    # full_time, full_data = complete_data[0:1, :, 2:3], complete_data[1:, :, 2:3]
    print('Full Time shape: {}, Full Data shape: {}'.format(full_time.shape, full_data.shape))
    data_size = full_data.shape[1]
    normalized, scalars = normalize_data(full_data)
    # unnormalized = denormalize_data(normalized, scalars)
    # diff = np.abs(unnormalized - full_data)
    # print('Absolute Differences: min - {} max - {} total - {}'.format(diff.min(), diff.max(), diff.sum()))
    if mode == 'standard':
        if conditional:
            mlb = MultiLabelBinarizer()
            new_labels = mlb.fit_transform(labels)
            print('Classes: {}'.format(mlb.classes_))
            print('labels shape: {}, new labels shape: {}'.format(labels.shape, new_labels.shape))
            normalized = normalized.repeat(1e3, axis=0)  # 1e4
            new_labels = new_labels.repeat(1e3, axis=0)
            num_tests = 4
            discriminator = make_conditional_af_accel_discriminator(data_size, num_tests)
            generator = make_conditional_af_accel_generator(latent_dimension, data_size, num_tests)
            cwgan = cWGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dimension)
            cwgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                          g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
                          metrics=[metric_fft_score, 'accuracy']
                          )
            cwgan.set_train_epochs(5, 1)
            temp_label = mlb.transform([(1, 3)])
            generator.predict((tf.zeros(shape=(1, latent_dimension)), tf.constant(temp_label)))
            early_stop = EarlyStopping(monitor='metric_fft_score', mode='min', min_delta=1e-2, verbose=1, patience=5,
                                       restore_best_weights=True)
            print('FFT Score before training: {}'.format(get_fft_score(normalized[0:128], generator.predict(
                (tf.random.normal(shape=(1, latent_dimension)),
                 tf.constant(mlb.transform([[1, 3]]), dtype=data_type))))))  # time,
            cwgan.fit((normalized, new_labels), epochs=epochs, batch_size=batch_size,
                      callbacks=[fft_callback(), early_stop])
            # cwgan.fit(x=benign_data, y=labels, epochs=epochs, batch_size=batch_size, callbacks=callback_list)
            # generator.save('./models/conditional_af_accel_generator')
            # discriminator.save('./models/conditional_af_accel_discriminator')
            rp = RecurrencePlot()
            eval_size = 64
            eval_labels = np.random.randint(0, 1, (eval_size, num_tests))
            prediction = generator.predict((tf.random.normal(shape=(eval_size, latent_dimension)), eval_labels))
            # recurrence difference plot
            plot_recurrence_diff(normalized[eval_labels[0]], prediction[0])
            # plt.show()
            plot_correlations(normalized[eval_labels[0]], prediction[0])
            # plt.show()
            print(get_cross_correlate_score(full_data[0:eval_size], prediction))
            print(get_fft_score(normalized[0:128], prediction[0:64]))  # time,
            for idx in range(num_tests):
                print('Current test: {}'.format(idx + 1))
                prediction = generator.predict((tf.random.normal(shape=(1, latent_dimension)),
                                                tf.constant(mlb.transform([[idx + 1]]), dtype=data_type)))
                plot_data(full_time[0], prediction,
                          normalized[idx + 1], show=False, save=False,
                          save_path='./results/AF_5_23_21_')
            prediction = generator.predict((tf.random.normal(shape=(1, latent_dimension)),
                                            tf.constant(mlb.transform([[1, 3]]), dtype=data_type)))
            plot_data(full_time[0], prediction,
                      normalized[idx + 1], show=False, save=False,
                      save_path='./results/AF_5_23_21_')

            plt.show()
        else:
            normalized = normalized.repeat(3e3, axis=0)  # 1e4
            print('Normalized shape: {}'.format(normalized.shape))
            dataset = data_to_dataset(normalized, dtype=data_type, batch_size=batch_size, shuffle=True)

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
            callback_list = [fft_callback(), early_stop, print_logs_callback(), csv_log, tb]
            print('FFT Score before training: {}'.format(get_fft_score(normalized[0:128], generator.predict(
                tf.random.normal(shape=(batch_size, latent_dimension))))))  # time,
            print(average_wasserstein(normalized[0:2], generator.predict(
                tf.random.normal(shape=(2, latent_dimension)))[0:2]))
            plot_wasserstein_histogram([discriminator.predict(normalized[0:full_data.shape[0]]),
                                        discriminator.predict(generator.predict(
                                            tf.random.normal(shape=(64, latent_dimension))))])
            wgan.fit(dataset, epochs=epochs, batch_size=batch_size, callbacks=callback_list)
            plot_wasserstein_histogram([discriminator.predict(normalized[0:full_data.shape[0]]),
                                        discriminator.predict(generator.predict(
                                            tf.random.normal(shape=(64, latent_dimension))))])
            plt.show()
            # Saving models
            # generator.save('models/af_accel_generator_full_new')
            # discriminator.save('models/af_accel_discriminator_full_new')
            rp = RecurrencePlot()
            prediction = generator.predict(tf.random.normal(shape=(64, latent_dimension)))
            # recurrence difference plot
            plot_recurrence_diff(normalized[0], prediction[0])
            # plt.show()
            plot_correlations(normalized[0], prediction[0])
            # plt.show()
            print(get_cross_correlate_score(normalized[0], prediction[0]))
            print('FFT Score after training: {}'.format(
                get_fft_score(normalized[0:128], prediction[0:batch_size])))  # time,
            plot_data(full_time[0], prediction[0:4], normalized[0:4], show=True, save=False,
                      save_path='./results/AF_5_23_21_')
    if mode == 'ebgan':
        norm_repeat = normalized.repeat(3e3, axis=0)  # 1e4
        ae_early_stop = EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-8, verbose=1, patience=3,
                                       restore_best_weights=True)
        eb_early_stop = EarlyStopping(monitor='Reconstruction error', mode='min', min_delta=1e-12, verbose=1, patience=3,
                                       restore_best_weights=True)
        # Make autoencoder
        autoencoder = make_fcc_autoencoder(data_size, 64)
        dataset = data_to_dataset(normalized)
        autoencoder.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam')
        autoencoder.fit(norm_repeat, norm_repeat, epochs=256, batch_size=batch_size, validation_split=0.2,
                        callbacks=[ae_early_stop])
        plot_data(full_time[0], autoencoder.predict(normalized)[0:6], normalized[0:6], show=True, save=False,
                  save_path='./results/AF_5_23_21_')


        # Use autoencoder loss as gan loss function
        generator = make_af_accel_generator(latent_dimension, data_size, data_type=data_type)
        # generator = make_af_accel_fcc_generator(latent_dimension, data_size, data_type=data_type)
        ebgan = EBGAN(discriminator=autoencoder, generator=generator, latent_dim=latent_dimension)
        ebgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                    g_optimizer=keras.optimizers.Adam(learning_rate=0.0008),
                    d_loss_fn=tf.keras.losses.mean_squared_error,
                    g_loss_fn=tf.keras.losses.mean_squared_error,
                     metrics=[metric_fft_score, tf_avg_wasserstein]
                     )
        ebgan.fit(norm_repeat, epochs=2, batch_size=batch_size, shuffle=True) # , callbacks=[eb_early_stop]
        # Saving models
        # generator.save('models/af_accel_generator_full')
        # discriminator.save('models/af_accel_discriminator_full')
        rp = RecurrencePlot()
        prediction = generator.predict(tf.random.normal(shape=(64, latent_dimension)))
        # recurrence difference plot
        plot_recurrence_diff(normalized[0], prediction[0])
        # plt.show()
        plot_correlations(normalized[0], prediction[0])
        # plt.show()
        print('Cross Correlation Score: {}'.format(get_cross_correlate_score(normalized[0], prediction[0])))
        print('FFT Score after training: {}'.format(
            get_fft_score(normalized[0:128], prediction[0:batch_size])))  # time,
        plot_data(full_time[0], prediction[0:8], autoencoder.predict(prediction[0:8]), show=True, save=False,
                  save_path='./results/AF_5_23_21_')
