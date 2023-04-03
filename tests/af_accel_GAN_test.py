import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from src import af_accel_GAN
from src.af_accel_GAN import prepare_data, plot_power_spectrum, plot_spectrogram, \
    plot_wasserstein_histogram
from src.model_architectures.af_accel_GAN_architecture import (
    make_af_accel_discriminator, make_af_accel_generator,
    make_conditional_af_accel_discriminator,
    make_conditional_af_accel_generator, make_af_accel_fcc_generator)
from src.AF_gan import get_fft_score
from src.sine_gan import generate_sine


class AfAccelGANArchitectureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.latent_dim = 24
        self.vector_size = 5_000

    def test_fcc_generator_succeeds(self):
        model = make_af_accel_fcc_generator(
            self.latent_dim, self.vector_size, summary=True)
        self.assertTrue(bool(model))

    def test_discriminator_succeeds(self):
        model = make_af_accel_discriminator(self.vector_size, summary=True)
        self.assertTrue(bool(model))

    def test_generator_succeeds(self):
        model = make_af_accel_generator(
            self.latent_dim, self.vector_size, summary=True)
        self.assertTrue(bool(model))
        self.assertEqual(model.output_shape, (None, self.vector_size))

    def test_conditional_discriminator_succeeds(self):
        model = make_conditional_af_accel_discriminator(
            self.vector_size, self.num_freq, summary=True)
        self.assertTrue(bool(model))

    def test_conditional_generator_succeeds(self):
        model = make_conditional_af_accel_generator(
            self.latent_dim, self.vector_size, self.num_freq, summary=True)
        self.assertTrue(bool(model))
        self.assertEqual(model.output_shape, (None, self.vector_size))

    @patch('af_accel_GAN.np', spec=True)
    def test_load_data_time_separate(self, mock_load):
        mock_load.loadtxt.return_value = np.asarray([[0, 0, 1, 0], [1, 0, 1, 0],
                                                     [0, 0, 0, 1], [1, 1, 1, 1]])
        time, data = af_accel_GAN.load_data('fake_path.txt', True)
        print(dir(mock_load))
        mock_load.loadtxt.assert_called_with('fake_path.txt', delimiter=',', skiprows=2)
        self.assertTrue(np.array_equal(time, [0, 1, 0, 1]))
        self.assertTrue(np.array_equal(data, [[0, 1, 0], [0, 1, 0],
                                              [0, 0, 1], [1, 1, 1]]))

    @patch('af_accel_GAN.np', spec=True)
    def test_load_data_time_not_separate(self, mock_load):
        mock_load.loadtxt.return_value = np.asarray([[0, 0, 1], [1, 0, 1],
                                                     [0, 0, 0], [1, 1, 1]])
        full = af_accel_GAN.load_data('fake_path.txt', False)
        mock_load.loadtxt.assert_called_with('fake_path.txt', delimiter=',', skiprows=2)
        self.assertTrue(np.array_equal(full, [[0, 0, 1], [1, 0, 1],
                                              [0, 0, 0], [1, 1, 1]]))
        print(full.shape)

    def test_fft_score(self):
        wave = generate_sine(0, 5, self.vector_size, 2)
        print(get_fft_score(np.array([wave]), np.array([wave])))
        self.assertAlmostEqual(
            get_fft_score(np.array([wave]), np.array([wave])), 0, 8)

class AfAccelGANTest(unittest.TestCase):
    def setUp(self) -> None:
        self.latent_dim = 24
        self.vector_size = 5_000
        self.num_freq = 5

    def load_data(self):
        self.skipTest()

    def load_data_files(self):
        self.skipTest()

    def test_plot_power_spectrum(self):
        start, end = 0, 5
        fs = self.vector_size / (end - start)
        wave = generate_sine(start, end, self.vector_size, 2)
        wave2 = generate_sine(start, end, self.vector_size, 2, 4)
        waves = np.asarray((wave, wave2))
        plot_power_spectrum(wave, fs)
        plot_power_spectrum(waves, fs)

    def test_plot_spectrogram(self):
        start, end = 0, 5
        fs = self.vector_size / (end - start)
        wave = generate_sine(start, end, self.vector_size, 2)
        wave2 = generate_sine(start, end, self.vector_size, 2, 4)
        waves = np.asarray((wave, wave2))
        plot_spectrogram(wave, fs)
        plot_spectrogram(waves, fs)

    def test_prepare_data_succeeds(self):
        # Mimic rough idea of shape for getting data, multiply so norm is different
        complete_data = np.ones(shape=(4, 20, 7)) * 3
        # Time is all zeros now
        complete_data[:, :, 0] = 0
        prepared_data = prepare_data(
            complete_data, scaling='normalize', return_labels=True)
        np.testing.assert_array_equal(prepared_data['data'], np.ones(shape=(17, 20))*3)
        self.assertTrue(np.array_equal(prepared_data['times'], np.zeros(shape=(4, 20))))
        self.assertTrue('labels' in prepared_data)
        self.assertTrue(np.array_equal(prepared_data['labels'], [[1], [2], [3], [4]]*4))
        # self.assertTrue(np.array_equal(normalized, prepared_data['normalized']))
        # self.assertTrue(np.array_equal(scalars, prepared_data['scalars']))


    def test_average_wasserstein(self):
        self.skipTest()

    def test_tf_avg_wasserstein(self):
        self.skipTest()

    def test_fft_score(self):
        wave = generate_sine(0, 5, self.vector_size, 2)
        print(get_fft_score(np.array([wave]), np.array([wave])))
        self.assertAlmostEqual(
            get_fft_score(np.array([wave]), np.array([wave])), 0, 8)

    def test_fft_score_one_dimension(self):
        wave = np.array(generate_sine(0, 10, 4_000, 1, 4))
        print(get_fft_score(wave, wave))
        self.assertAlmostEqual(get_fft_score(wave, wave), 0, 8)


class AfAccelGANModelTest(unittest.TestCase):
    def test_standard_conditional(self):
        self.skipTest()

    def test_plot_wasserstein_histogram(self):
        data = np.random.normal(0, 1, (2, 1024))
        fig_num = plot_wasserstein_histogram(data)
        self.assertTrue(fig_num)
        plt.show()


if __name__ == '__main__':
    unittest.main()
