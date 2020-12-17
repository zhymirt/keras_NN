""" GAN Class taken from https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit"""
import math
from random import randint, random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
from torch import sin


def discrim_wasserstein_loss_fn(y_true, y_pred):
    difference = y_true - y_pred # tf.subtract(y_true, y_pred)
    return -tf.reduce_mean(difference, axis=-1)

def gen_wasserstein_loss_fn(y_true, y_pred):
    return -tf.reduce_mean(y_pred, axis=-1)

def generate_sine(start, end, points, amplitude=1, frequency=1):
    time = np.linspace(0, 2, 100)
    signal = amplitude*np.sin(2*np.pi*frequency*time)
    return signal

def plot_data(data):
    plt.plot(np.linspace(0, 100, num=100), data)

class DiscriminatorWassersteinLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return discrim_wasserstein_loss_fn(y_true, y_pred)
        # y_pred = tf.convert_to_tensor_v2(y_pred)
        # y_true = tf.cast(y_true, y_pred.dtype)
        # return tf.reduce_mean(y_pred - y_true, axis=-1)

class GeneratorWassersteinLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return gen_wasserstein_loss_fn(y_true, y_pred)
        # y_pred = tf.convert_to_tensor_v2(y_pred)
        # y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(y_pred, axis=-1)

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        data_type = real_images.dtype
        print(tf.shape(real_images)[0])
        print(real_images.shape[0])
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        print(random_latent_vectors)
        generated = self.generator(random_latent_vectors)
        print(generated)
        combined = tf.concat([real_images, generated], axis=0)
        # fakes_labels_temp = tf.zeros((batch_size, 1))
        # print(fakes_labels_temp)
        # fakes_labels = keras.utils.to_categorical(fakes_labels_temp, num_classes=2)
        # fakes_labels, real_labels = np.array([[1, 0] for _ in range(batch_size)], dtype=data_type), np.array([[0, 1] for _ in range(batch_size)], dtype=data_type)
        # labels = tf.concat((fakes_labels, real_labels))

        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        fakes_labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            # d_loss = self.d_loss_fn(self.discriminator(real_images), self.discriminator(generated))
            d_loss = self.d_loss_fn(labels, self.discriminator(combined))
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            # g_loss = self.g_loss_fn(None, predictions)
            g_loss = self.g_loss_fn(fakes_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {'d_loss': d_loss, 'g_loss': g_loss}

if __name__ == '__main__':
    # discriminator = keras.Sequential(
    # [
    #     keras.Input(shape=(28, 28, 1)),
    #     layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.GlobalMaxPooling2D(),
    #     layers.Dense(1),
    # ],
    # name="discriminator",
    # )

    # # Create the generator
    # latent_dim = 128
    # generator = keras.Sequential(
    #     [
    #         keras.Input(shape=(latent_dim,)),
    #         # We want to generate 128 coefficients to reshape into a 7x7x128 map
    #         layers.Dense(7 * 7 * 128),
    #         layers.LeakyReLU(alpha=0.2),
    #         layers.Reshape((7, 7, 128)),
    #         layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
    #         layers.LeakyReLU(alpha=0.2),
    #         layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
    #         layers.LeakyReLU(alpha=0.2),
    #         layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    #     ],
    #     name="generator",
    # )
    # batch_size = 64
    # (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    # all_digits = np.concatenate([x_train, x_test])
    # all_digits = all_digits.astype("float32") / 255.0
    # all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
    # dataset = tf.data.Dataset.from_tensor_slices(all_digits)
    # dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    # gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    # gan.compile(
    #     d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    #     g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    #     d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    #     g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
    # )

    # # To limit the execution time, we only train on 100 batches. You can train on
    # # the entire dataset. You will need about 20 epochs to get nice results.
    # gan.fit(dataset.take(100), epochs=1)

    latent_dimension = 200
    vector_size = 100
    # create discriminator and generator
    discrim = keras.Sequential(
        [
            Dense(10, input_shape=(vector_size,), activation='relu'),
            Dense(100, activation='relu'),
            Dense(1)
        ]
    )
    # discrim.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True))
    generator = keras.Sequential([
        Dense(100, activation='relu', input_shape=(latent_dimension,)),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(40, activation='relu'),
        Dense(vector_size, activation='relu')
    ])
    # generator.compile(optimizer='adam', loss='mean_squared_logarithmic_error')
    # generator attempts to produce even numbers, discriminator will tell if true or not
    data_type = 'float32'

    range_min, random_range, data_size = 0, 50, 720
    even_min, even_range = range_min, int(random_range / 2)
    trained, passes, min_passes = False, 0, 3
    epoch, epochs = 0, 10
    batch_size = 64
    label_alias = {'fake': 0, 'real': 1}
    starts = [randint(0, 100)/100 for _ in range(1000)] # generate n beginning data points
    benign_data = [generate_sine(val, val + 1, 100) for val in starts] # generate 100 points of sine wave
    # benign_data = [[math.sin(val)] for val in range(data_size)] # for sine numbers
    # print('Benign data: {}'.format(benign_data))
    # benign_data = np.array([[randint(range_min, random_range)*2 + random()] for _ in range(data_size)], dtype=data_type)
    dataset = tf.data.Dataset.from_tensor_slices(tf.cast(benign_data, dtype=data_type))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    print('Dataset: {}'.format(dataset.take(10)))
    gan = GAN(discriminator=discrim, generator=generator, latent_dim=latent_dimension)
    gan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                # g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                # d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
                g_loss_fn=GeneratorWassersteinLoss(),
                d_loss_fn=DiscriminatorWassersteinLoss()
    )
    gan.fit(dataset, epochs=20)
    for num in range(-20, 21):
        random_latent_vectors = tf.random.normal(shape=(1, latent_dimension))
        print('Value at {}: {}'.format(num, generator.predict(random_latent_vectors)))
