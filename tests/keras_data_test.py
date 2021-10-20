import unittest
from unittest import TestCase

import numpy as np
from numpy import sqrt

from keras_data import abs_mean, frequency_center, get_fft_freq, get_fft, rms, root_mean_square_frequency, \
    root_variance_frequency
from sine_gan import generate_sine
from matplotlib import pyplot as plt


class StatisticsTestCase(TestCase):
    def test_std_dev(self):
        self.skipTest()

    def test_standardize(self):
        self.skipTest()


class MyTestCase(TestCase):
    def setUp(self) -> None:
        self.start_time, self.stop_time = 0, 5
        self.test_vector_size = 5_000
        self.time_step = (self.stop_time - self.start_time) / self.test_vector_size
        self.time = np.linspace(self.start_time, self.stop_time, self.test_vector_size)
        self.places = 1
        # self.test_vector = list()

    def test_abs_mean(self):
        self.skipTest('not written yet')

    def test_rms(self):
        # precalculate root inverse, place values for almost equal
        root_2, places = 1 / sqrt(2), 2
        # rms for constant amplitude should be amplitude
        for idx in range(20):
            with self.subTest():
                dc_wave = np.asarray([idx for _ in range(self.test_vector_size)])
                self.assertEqual(rms(dc_wave), idx)
        # rms for sine wave should be amplitude/root of 2
        for idx in range(20):
            with self.subTest():
                wave = generate_sine(
                    self.start_time, self.stop_time,
                    self.test_vector_size, amplitude=idx)
                self.assertAlmostEqual(rms(wave), idx * root_2, places=places)
        # rms for sine wave should be amplitude/root of 2,
        # no matter frequency
        for idx in range(1, 21):
            with self.subTest():
                wave = generate_sine(
                    self.start_time, self.stop_time,
                    self.test_vector_size, frequency=idx)
                self.assertAlmostEqual(rms(wave), root_2, places=places)

    def test_skewness(self):
        self.skipTest('not written yet')

    def test_kurtosis(self):
        self.skipTest('not written yet')

    def test_crest_factor(self):
        self.skipTest('not written yet')

    def test_shape_factor(self):
        self.skipTest('not written yet')

    def test_impulse_factor(self):
        self.skipTest('not written yet')

    def test_get_fft(self):
        # self.skipTest('Analyze plot manually.')
        wave = generate_sine(self.start_time, self.stop_time, self.test_vector_size, amplitude=1, frequency=1)
        plt.figure()
        plt.plot(get_fft_freq(wave, self.time_step), get_fft(wave))
        plt.show()

    def test_get_fft_freq(self):
        self.skipTest('Unnecessary for now.')

    def test_frequency_center(self):
        # self.skipTest('not written yet')
        places = 2
        all_zeros = np.asarray([0 for _ in range(self.test_vector_size)])
        self.assertTrue(np.isnan(frequency_center(all_zeros, time_step=self.time_step)),
                        'frequency center for vector of zeros should be nan.')
        # with self.assertRaises(BaseException):
        #     fail_test = frequency_center(all_zeros, timestep=self.time_step)
        #     print("fail_test result {}".format(fail_test))
        for idx in range(1, 21):
            with self.subTest():
                wave = generate_sine(
                    self.start_time, self.stop_time,
                    self.test_vector_size, amplitude=idx)
                self.assertAlmostEqual(frequency_center(wave, self.time_step), 1, places=places)
        for idx in range(1, 21):
            with self.subTest():
                wave = generate_sine(
                    self.start_time, self.stop_time,
                    self.test_vector_size, frequency=idx)
                self.assertAlmostEqual(frequency_center(wave, self.time_step), idx, places=places)
        wave = generate_sine(self.start_time, self.stop_time, self.test_vector_size, amplitude=20, frequency=100)
        self.assertAlmostEqual(frequency_center(wave, time_step=self.time_step), 100, self.places)

    def test_root_mean_square_frequency(self):
        # self.skipTest('not written yet')
        root_2, places = 1 / sqrt(2), 2
        # # no power in frequencies means trouble
        # for idx in range(20):
        #     with self.subTest():
        #         dc_wave = np.asarray([idx for _ in range(
        #             self.test_vector_size)])
        #         self.assertEqual(
        #             root_mean_square_frequency(dc_wave, self.time_step), 0)
        # rmsf should be amplitude invariant
        for idx in range(1, 21):
            with self.subTest():
                wave = generate_sine(
                    self.start_time, self.stop_time, self.test_vector_size,
                    amplitude=idx)
                self.assertAlmostEqual(
                    root_mean_square_frequency(
                        wave, self.time_step), 1, places=places)
        # rmsf should be almost equal to frequency
        for idx in range(1, 21):
            with self.subTest():
                wave = generate_sine(
                    self.start_time, self.stop_time, self.test_vector_size,
                    frequency=idx)
                self.assertAlmostEqual(
                    root_mean_square_frequency(
                        wave, self.time_step), idx, places=places)

    def test_root_variance_frequency(self):
        self.skipTest('not finished yet')
        places = 2
        # rmsf should be amplitude invariant
        # for idx in range(1, 21):
        #     with self.subTest():
        #         wave = generate_sine(
        #             self.start_time, self.stop_time, self.test_vector_size,
        #             amplitude=idx)
        #         self.assertAlmostEqual(
        #             root_variance_frequency(
        #                 wave, self.time_step), 0, places=places)
        # rmsf should be almost equal to frequency
        # for idx in range(1, 21):
        #     with self.subTest():
        #         wave = generate_sine(
        #             self.start_time, self.stop_time, self.test_vector_size,
        #             frequency=idx)
        #         self.assertAlmostEqual(
        #             root_variance_frequency(
        #                 wave, self.time_step), 0, places=places)

    # def tearDown(self) -> None:
    #     self.test_vector.clear()


if __name__ == '__main__':
    unittest.main()
