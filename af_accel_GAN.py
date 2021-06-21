import os
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from pyts.image import RecurrencePlot
from tensorflow import keras

from AF_gan import normalize_data, denormalize_data, metric_fft_score, get_fft_score, get_cross_correlate_score, \
    plot_data, plot_correlations, plot_recurrence_diff
from keras_data import data_to_dataset
from keras_gan import WGAN, fft_callback, cWGAN
from model_architectures.af_accel_GAN_architecture import make_af_accel_discriminator, make_af_accel_generator, \
    make_conditional_af_accel_discriminator, make_conditional_af_accel_generator


def load_data(filename, separate_time=True):
    fn_data = np.loadtxt(filename, delimiter=',', skiprows=2)
    if separate_time:
        return fn_data[:, 0], fn_data[:, 1:]
    else:
        return fn_data


def load_data_files(filenames, separate_time=True):
    fn_data = np.array([load_data(name, separate_time=False) for name in filenames])
    if separate_time:
        return fn_data[:, :, 0], fn_data[:, :, 1:]
    else:
        return fn_data


def prepare_data(complete, scaling=None, return_labels=False):
    returned_values = dict()
    print('Complete shape: {}'.format(complete.shape))
    full_time = complete[:, :, 0]
    full_data, labels = [], []
    print('Full time shape: {}'.format(full_time.shape))
    for example_set in complete.transpose((0, 2, 1)):
        for test_num, test in enumerate(example_set[1:]):
            if np.sum(np.square(test)) > 1e-8: # numbers aren't all zero
                # print('Test #{} shape: {}'.format(test_num + 1, test.shape))
                labels.append(test_num + 1)
                full_data.append(test)
    full_data, labels = np.array(full_data), np.array(labels)
    returned_values['times'] = full_time
    returned_values['data'] = full_data
    if return_labels:
        returned_values['labels'] = labels
    if scaling is not None and scaling == 'normalize':
        returned_values['normalized'], returned_values['scalars'] = normalize_data(full_data)
    return returned_values

if __name__=='__main__':
    latent_dimension, data_type, epochs, batch_size, conditional = 256, 'float32', 32, 16, True
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
    for example_set in complete_data.transpose((0, 2, 1)):
        for test_num, test in enumerate(example_set[1:]):
            if np.sum(np.square(test)) > 1e-8: # numbers aren't all zero
                # print('Test #{} shape: {}'.format(test_num + 1, test.shape))
                labels.append(test_num + 1)
                full_data.append(test)
    #             plt.figure()
    #             plt.plot(full_time[0], test)
    #             plt.title('Plot for test #{}'.format(test_num))
    # plt.show()
    full_data, labels = np.array(full_data), np.array(labels)
    # exit()
    # print('Complete shape: {}'.format(complete_data.shape))
    # full_time, full_data = complete_data[0:1, :, 2:3], complete_data[1:, :, 2:3]
    print('Full Time shape: {}, Full Data shape: {}'.format(full_time.shape, full_data.shape))
    data_size = full_data.shape[1]
    normalized, scalars = normalize_data(full_data)
    # unnormalized = denormalize_data(normalized, scalars)
    # diff = np.abs(unnormalized - full_data)
    # print('Absolute Differences: min - {} max - {} total - {}'.format(diff.min(), diff.max(), diff.sum()))
    if conditional:
        normalized = normalized.repeat(1e3, axis=0)  # 1e4
        labels = labels.repeat(1e3, axis=0)
        num_frequencies = 6
        discriminator = make_conditional_af_accel_discriminator(data_size, num_frequencies)
        generator = make_conditional_af_accel_generator(latent_dimension, data_size, num_frequencies)
        cwgan = cWGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dimension)
        cwgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      g_optimizer=keras.optimizers.Adam(learning_rate=0.0006),
                      metrics=[metric_fft_score, 'accuracy']
                      )
        cwgan.set_train_epochs(5, 1)
        generator.predict((tf.zeros(shape=(1, latent_dimension)), tf.constant([1])))
        early_stop = EarlyStopping(monitor='metric_fft_score', mode='min', min_delta=1e-2, verbose=1, patience=5)
        cwgan.fit((normalized, labels), epochs=epochs, batch_size=batch_size, callbacks=[fft_callback(), early_stop])
        # cwgan.fit(x=benign_data, y=labels, epochs=epochs, batch_size=batch_size, callbacks=callback_list)
        generator.save('./models/conditional_af_accel_generator')
        discriminator.save('./models/conditional_af_accel_discriminator')
        rp = RecurrencePlot()
        eval_size = 64
        eval_labels = np.random.choice(labels, eval_size)
        prediction = generator.predict((tf.random.normal(shape=(eval_size, latent_dimension)), eval_labels))
        # recurrence difference plot
        plot_recurrence_diff(normalized[eval_labels[0]], prediction[0])
        # plt.show()
        plot_correlations(normalized[eval_labels[0]], prediction[0])
        # plt.show()
        print(get_cross_correlate_score(full_data, prediction[0]))
        print(get_fft_score(normalized[0:128], prediction[0:64]))  # time,
        for idx in range(num_frequencies - 1):
            print('Current test: {}'.format(idx + 1))
            prediction = generator.predict((tf.random.normal(shape=(1, latent_dimension)), tf.constant([idx + 1], dtype=data_type)))
            plot_data(full_time[0], prediction,
                  normalized[idx + 1], show=False, save=False,
                  save_path='./results/AF_5_23_21_')
        plt.show()
    else:
        normalized = normalized.repeat(1e3, axis=0)  # 1e4
        print('Normalized shape: {}'.format(normalized.shape))
        dataset = data_to_dataset(normalized, dtype=data_type, batch_size=batch_size, shuffle=True)

        discriminator = make_af_accel_discriminator(data_size, data_type=data_type)
        generator = make_af_accel_generator(latent_dimension, data_size, data_type=data_type)
        wgan = WGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dimension)
        wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                     g_optimizer=keras.optimizers.Adam(learning_rate=0.008),
                     metrics=[metric_fft_score, 'accuracy']
                     )
        wgan.set_train_epochs(4, 1)
        early_stop = EarlyStopping(monitor='metric_fft_score', mode='min', min_delta=1e-2, verbose=1, patience=7)
        print(get_fft_score(normalized[0:128], generator.predict(tf.random.normal(shape=(batch_size, latent_dimension)))))  # time,
        wgan.fit(dataset, epochs=epochs, batch_size=batch_size, callbacks=[fft_callback(), early_stop])
        # Saving models
        generator.save('models/af_accel_generator_full')
        discriminator.save('models/af_accel_discriminator_full')
        rp = RecurrencePlot()
        prediction = generator.predict(tf.random.normal(shape=(64, latent_dimension)))
        # recurrence difference plot
        plot_recurrence_diff(normalized[0], prediction[0])
        # plt.show()
        plot_correlations(normalized[0], prediction[0])
        # plt.show()
        print(get_cross_correlate_score(full_data, prediction[0]))
        print(get_fft_score(normalized[0:128], prediction[0:batch_size]))  # time,
        plot_data(full_time[0], prediction[0], normalized[0], show=True, save=False,
                  save_path='./results/AF_5_23_21_')
