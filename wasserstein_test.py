from unittest import TestCase
import unittest
import numpy as np
# import keras_NN.keras_gan
from keras_gan import discrim_wasserstein_loss_fn

class WassersteinTest(TestCase):

    # def test_Fail(self):
    #     self.assertEqual(False, True)

    def test_wasserstein_basic(self):
        item_1 = np.array([0.1, 0.2, 0.4, 0.3], dtype='float32')
        item_2 = np.array([0.2, 0.1, 0.2, 0.5], dtype='float32')
        wasserstein = np.around(discrim_wasserstein_loss_fn(item_1, item_2).numpy(), 8)
        self.assertEqual(wasserstein, np.array(0.6, dtype='float32'))

    def test_wasserstein_nested(self):
        item_1 = np.array([[0.1, 0.2, 0.4, 0.3]], dtype='float32')
        item_2 = np.array([[0.2, 0.1, 0.2, 0.5]], dtype='float32')
        wasserstein = np.around(discrim_wasserstein_loss_fn(item_1, item_2).numpy(), 8)
        self.assertEqual(wasserstein, np.array([0.6], dtype='float32'))
if __name__ == '__main__':
    unittest.main()