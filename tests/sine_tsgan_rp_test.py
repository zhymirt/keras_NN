import unittest
from unittest import TestCase

from src.model_architectures.sine_tsgan_rp_architecture import (
    make_sine_wgan_discriminator, make_sine_wgan_generator,
    make_spectrogram_discriminator, make_spectrogram_generator)

# from sine_tsgan_rp import (make_spectrogram_discriminator, make_spectrogram_generator,
# make_sine_wgan_discriminator, make_sine_wgan_generator)

class SineTSGANRPTest(TestCase):
    def test_spectrogram_discriminator_succeeds(self):
        model = make_spectrogram_discriminator(summary=True)
        self.assertTrue(bool(model))
    def test_spectrogram_generator_succeeds(self):
        model = make_spectrogram_generator(256, summary=True)
        self.assertTrue(bool(model))
    def test_sine_wgan_discriminator(self):
        model = make_sine_wgan_discriminator(100, summary=True)
        self.assertTrue(bool(model))
    def test_sine_wgan_generator(self):
        model = make_sine_wgan_generator(256, 100, summary=True)
        self.assertTrue(bool(model))

if __name__ == '__main__':
    unittest.main()