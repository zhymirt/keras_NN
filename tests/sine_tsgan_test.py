from unittest import TestCase

from src.model_architectures.sine_tsgan_architecture import make_sine_tsgan_discriminator_1, make_sine_tsgan_generator_1, \
    make_sine_tsgan_discriminator_2, make_sine_tsgan_generator_2


class SineTsGANTest(TestCase):

    def test_sine_tsgan_discriminator_1_succeeds(self):
        model = make_sine_tsgan_discriminator_1(summary=True)
        self.assertTrue(bool(model))
    def test_sine_tsgan_generator_1_succeeds(self):
        model = make_sine_tsgan_generator_1(256, summary=True)
        self.assertTrue(bool(model))
    def test_sine_tsgan_discriminator_2_succeeds(self):
        model = make_sine_tsgan_discriminator_2(100, summary=True)
        self.assertTrue(bool(model))
    def test_sine_tsgan_generator_2_succeeds(self):
        model = make_sine_tsgan_generator_2(256, 100, summary=True)
        self.assertTrue(bool(model))