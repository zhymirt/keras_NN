import unittest
from unittest import TestCase

from model_architectures.sine_gan_architecture import make_sine_gan_fcc_discriminator, make_sine_gan_fcc_generator, \
    make_sine_gan_cnn_discriminator, make_sine_gan_cnn_generator, make_conditional_sine_gan_cnn_discriminator, \
    make_conditional_sine_gan_cnn_generator


class SineGANTest(TestCase):
    def setUp(self) -> None:
        self.latent_dim = 256
        self.vector_size = 6_000
        self.num_freq = 4

    def test_sine_gan_fcc_discriminator_succeeds(self):
        model = make_sine_gan_fcc_discriminator(self.vector_size, summary=True)
        self.assertTrue(bool(model))

    def test_sine_gan_fcc_generator_succeeds(self):
        model = make_sine_gan_fcc_generator(256, self.vector_size, summary=True)
        self.assertTrue(bool(model))

    def test_sine_gan_cnn_discriminator_succeeds(self):
        model = make_sine_gan_cnn_discriminator(self.vector_size, summary=True)
        self.assertTrue(bool(model))

    def test_sine_gan_cnn_generator_succeeds(self):
        model = make_sine_gan_cnn_generator(self.latent_dim, summary=True)
        self.assertTrue(bool(model))
        self.assertEqual(model.output_shape, (None, self.vector_size))

    def test_conditional_sine_gan_discriminator_succeeds(self):
        model = make_conditional_sine_gan_cnn_discriminator(self.vector_size, self.num_freq, summary=True)
        self.assertTrue(bool(model))

    def test_conditional_sine_gan_generator_succeeds(self):
        model = make_conditional_sine_gan_cnn_generator(self.latent_dim, self.vector_size, self.num_freq, summary=True)
        self.assertTrue(bool(model))


if __name__ == '__main__':
    unittest.main()