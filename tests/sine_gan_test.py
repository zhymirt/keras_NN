import unittest
from unittest import TestCase

from model_architectures.sine_gan_architecture import make_sine_gan_fcc_discriminator, make_sine_gan_fcc_generator, \
    make_sine_gan_cnn_discriminator, make_sine_gan_cnn_generator, make_conditional_sine_gan_cnn_discriminator, \
    make_conditional_sine_gan_cnn_generator


class SineGANTest(TestCase):

    def test_sine_gan_fcc_discriminator_succeeds(self):
        model = make_sine_gan_fcc_discriminator(6000, summary=True)
        self.assertTrue(bool(model))

    def test_sine_gan_fcc_generator_succeeds(self):
        model = make_sine_gan_fcc_generator(256, 6000, summary=True)
        self.assertTrue(bool(model))

    def test_sine_gan_cnn_discriminator_succeeds(self):
        model = make_sine_gan_cnn_discriminator(6000, summary=True)
        self.assertTrue(bool(model))

    def test_sine_gan_cnn_generator_succeeds(self):
        model = make_sine_gan_cnn_generator(256, summary=True)
        self.assertTrue(bool(model))
        self.assertEqual(model.output_shape, (None, 6000))

    def test_conditional_sine_gan_discriminator_succeeds(self):
        model = make_conditional_sine_gan_cnn_discriminator(6000, 4, summary=True)
        self.assertTrue(bool(model))

    def test_conditional_sine_gan_generator_succeeds(self):
        model = make_conditional_sine_gan_cnn_generator(256, 6000, 4, summary=True)
        self.assertTrue(bool(model))


if __name__ == '__main__':
    unittest.main()