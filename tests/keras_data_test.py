import unittest
from unittest import TestCase

import numpy as np

from keras_data import abs_mean, frequency_center, get_fft_freq, get_fft
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
        self.skipTest()

    def test_rms(self):
        self.skipTest()

    def test_skewness(self):
        self.skipTest()

    def test_kurtosis(self):
        self.skipTest()

    def test_crest_factor(self):
        self.skipTest()

    def test_shape_factor(self):
        self.skipTest()

    def test_impulse_factor(self):
        self.skipTest()

    def test_get_fft(self):
        # self.skipTest('Analyze plot manually.')
        wave = generate_sine(self.start_time, self.stop_time, self.test_vector_size, amplitude=1, frequency=1)
        plt.figure()
        plt.plot(get_fft_freq(wave, self.time_step), get_fft(wave))
        plt.show()

    def test_get_fft_freq(self):
        self.skipTest('Unnecessary for now.')

    def test_frequency_center(self):
        # self.skipTest()
        all_zeros = np.asarray([0 for _ in range(self.test_vector_size)])
        # self.assertTrue(np.isnan(frequency_center(all_zeros, timestep=self.time_step)),
        #                 'frequency center for vector of zeros should be nan.')
        # with self.assertRaises(BaseException):
        #     fail_test = frequency_center(all_zeros, timestep=self.time_step)
        #     print("fail_test result {}".format(fail_test))
        wave = generate_sine(self.start_time, self.stop_time, self.test_vector_size)
        # plt.figure()
        # plt.plot(self.time, wave)
        # plt.show()
        self.assertAlmostEqual(frequency_center(wave, time_step=self.time_step), 1, self.places)
        wave = generate_sine(self.start_time, self.stop_time, self.test_vector_size, amplitude=20)
        self.assertAlmostEqual(frequency_center(wave, time_step=self.time_step), 1, self.places)
        wave = generate_sine(self.start_time, self.stop_time, self.test_vector_size, amplitude=20, frequency=100)
        self.assertAlmostEqual(frequency_center(wave, time_step=self.time_step), 100, self.places)

    def test_root_mean_square(self):
        self.skipTest()

    def test_root_variance_frequency(self):
        self.skipTest()
    # def test_something(self):
    #     self.assertEqual(True, False)

    # def tearDown(self) -> None:
    #     self.test_vector.clear()



if __name__ == '__main__':
    unittest.main()

