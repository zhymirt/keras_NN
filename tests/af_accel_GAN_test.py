import os
import unittest

import numpy as np

from af_accel_GAN import load_data_files, prepare_data
from model_architectures.af_accel_GAN_architecture import make_af_accel_discriminator, make_af_accel_generator, \
    make_conditional_af_accel_discriminator, make_conditional_af_accel_generator, make_af_accel_fcc_generator
from AF_gan import get_fft_score, normalize_data
from keras_gan import generate_sine


class AfAccelGANTest(unittest.TestCase):
    def setUp(self) -> None:
        self.latent_dim = 10
        self.vector_size = 5_000
        self.num_freq = 5

    def test_fcc_generator_succeeds(self):
        model = make_af_accel_fcc_generator(self.latent_dim, self.vector_size, summary=True)
        self.assertTrue(bool(model))

    def test_discriminator_succeeds(self):
        model = make_af_accel_discriminator(self.vector_size, summary=True)
        self.assertTrue(bool(model))

    def test_generator_succeeds(self):
        model = make_af_accel_generator(self.latent_dim, self.vector_size, summary=True)
        self.assertTrue(bool(model))
        self.assertEqual(model.output_shape, (None, self.vector_size))

    def test_conditional_discriminator_succeeds(self):
        model = make_conditional_af_accel_discriminator(self.vector_size, self.num_freq, summary=True)
        self.assertTrue(bool(model))

    def test_conditional_generator_succeeds(self):
        model = make_conditional_af_accel_generator(self.latent_dim, self.vector_size, self.num_freq, summary=True)
        self.assertTrue(bool(model))
        self.assertEqual(model.output_shape, (None, self.vector_size))

    def test_fft_score(self):
        wave = generate_sine(0, 5, self.vector_size, 2)
        print(get_fft_score(np.array([wave]), np.array([wave])))
        self.assertAlmostEqual(get_fft_score(np.array([wave]), np.array([wave])), 0, 8)

    def test_fft_score_one_dimension(self):
        wave = np.array(generate_sine(0, 10, 4_000, 1, 4))
        print(get_fft_score(wave, wave))
        self.assertAlmostEqual(get_fft_score(wave, wave), 0, 8)

    def test_prepare_data_succeeds(self):
        complete_data = load_data_files([os.path.join('../../acceleration_data', name) for name in ('accel_1.csv',
                                                                                                    'accel_2.csv',
                                                                                                    'accel_3.csv',
                                                                                                    'accel_4.csv')],
                                        separate_time=False)
        full_time = complete_data[:, :, 0]
        full_data, labels = [], []
        print('Full time shape: {}'.format(full_time.shape))
        for example_set in complete_data.transpose((0, 2, 1)):
            for test_num, test in enumerate(example_set[1:5]):
                # print('Test #{} shape: {}'.format(test_num + 1, test.shape))
                labels.append(test_num + 1)
                full_data.append(test)
        full_data, labels = np.array(full_data), np.array(labels)
        # exit()
        # print('Complete shape: {}'.format(complete_data.shape))
        # full_time, full_data = complete_data[0:1, :, 2:3], complete_data[1:, :, 2:3]
        print('Full Time shape: {}, Full Data shape: {}'.format(full_time.shape, full_data.shape))
        data_size = full_data.shape[1]
        normalized, scalars = normalize_data(full_data)
        prepared_data = prepare_data(complete_data, scaling='normalize', return_labels=True)
        self.assertTrue(np.array_equal(full_data, prepared_data['data']))
        self.assertTrue(np.array_equal(full_time, prepared_data['times']))
        # self.assertTrue(np.array_equal(labels, prepared_data['labels']))
        # self.assertTrue(np.array_equal(normalized, prepared_data['normalized']))
        # self.assertTrue(np.array_equal(scalars, prepared_data['scalars']))


if __name__ == '__main__':
    unittest.main()
