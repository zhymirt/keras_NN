import unittest
from unittest import TestCase
from sine_tsgan_rp import (make_spectrogram_discriminator, make_spectrogram_generator,
make_sine_wgan_discriminator, make_sine_wgan_generator)

class SineTSGANRPTest(TestCase):
    def test_spectrogram_discriminator_succeeds(self):
        model = make_spectrogram_discriminator()
        self.assertTrue(bool(model))
    def test_spectrogram_generator_succeeds(self):
        model = make_spectrogram_generator(256)
        self.assertTrue(bool(model))
    def test_sine_wgan_discriminator(self):
        model = make_sine_wgan_discriminator(100)
        self.assertTrue(bool(model))
    def test_sine_wgan_generator(self):
        model = make_sine_wgan_generator(256, 100)
        self.assertTrue(bool(model))