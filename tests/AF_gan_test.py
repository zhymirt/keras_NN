import unittest
from unittest import TestCase

from src.model_architectures.AF_gan_architecture import (
    make_AF_discriminator, make_AF_generator, make_AF_rp_generator,
    make_AF_single_discriminator, make_AF_single_generator,
    make_AF_rp_discriminator, make_AF_spectrogram_discriminator_1,
    make_AF_spectrogram_generator_1)


class AFGANTest(TestCase):

    def test_discriminator_succeeds(self):
        model = make_AF_discriminator(5, 5501, summary=True)
        self.assertTrue(bool(model))

    def test_generator_succeeds(self):
        model = make_AF_generator(256, 5, 5501, summary=True)
        self.assertTrue(bool(model))
        self.assertEqual(model.output_shape, (None, 5501, 5))

    def test_AF_single_discriminator_succeeds(self):
        model = make_AF_single_discriminator(5501, summary=True)
        self.assertTrue(bool(model))

    def test_AF_single_generator_succeeds(self):
        model = make_AF_single_generator(256, 5501, summary=True)
        self.assertTrue(bool(model))
        self.assertEqual(model.output_shape, (None, 5501))

    def test_AF_rp_discriminator(self):
        model = make_AF_rp_discriminator(5501, summary=True)
        self.assertTrue(bool(model))
        self.assertEqual(model.input_shape, (None, 5501, 5501))

    def test_AF_rp_generator(self):
        model = make_AF_rp_generator(256, (5501, 5501), summary=True)
        self.assertTrue(bool(model))
        self.assertEqual(model.output_shape, (None, 5501, 5501))

    def test_conditional_discriminator_succeeds(self):
        self.skipTest('model architecture not made yet.')
        # model = make_conditional_discriminator(100, summary=True)
        # self.assertTrue(bool(model))

    def test_conditional_generator_succeeds(self):
        self.skipTest('model architecture not made yet.')
        # model = make_conditional_generator(256, 100, summary=True)
        # self.assertTrue(bool(model))

    def test_af_spectrogram_discriminator_succeeds(self):
        model = make_AF_spectrogram_discriminator_1(summary=True)
        self.assertTrue(bool(model))

    def test_af_spectrogram_generator_succeeds(self):
        model = make_AF_spectrogram_generator_1(256, summary=True)
        self.assertTrue(bool(model))
        self.assertEqual(model.output_shape, (None, 5, 129, 24))


if __name__ == '__main__':
    unittest.main()
