import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.losses import cosine_similarity

from custom_functions.custom_losses import wasserstein_loss_fn, wasserstein_metric_fn, critic_diversity_metric


class GAN(keras.Model):
    """ Class for generative adversarial network."""
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.discriminator_epochs = 1
        self.generator_epochs = 1
        self.d_optimizer = None
        self.g_optimizer = None
        self.d_loss_fn = None
        self.g_loss_fn = None

    def compile(
            self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, **kwargs):
        """ Compile model."""
        super(GAN, self).compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def set_train_epochs(self, discrim_epochs=1, gen_epochs=1):
        """ Set number of training iterations per model per epoch."""
        self.discriminator_epochs = max(1, int(discrim_epochs))
        self.generator_epochs = max(1, int(gen_epochs))

    def train_step(self, real_images):
        """ Training step iteration for model."""
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        data_type = real_images.dtype
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(
            batch_size, self.latent_dim))
        generated = self.generator(random_latent_vectors)
        # combined = tf.concat([real_images, generated], axis=0)
        #
        # labels = tf.concat(
        #     [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        # )
        fakes_labels = tf.ones((batch_size, 1))
        d_loss = self.train_discriminator(
            tf.ones(shape=(batch_size, 1)),
            tf.zeros(shape=(batch_size, 1)), real_images,
            generated, data_type=data_type)
        g_loss = self.train_generator(
            fakes_labels, random_latent_vectors, data_type=data_type)
        return {'d_loss': d_loss, 'g_loss': g_loss}

    def train_discriminator(
            self, real_labels, fake_labels, real_data, fake_data,
            data_type='float32'):
        """ Training step for discriminator."""
        d_loss = None
        combined_labels = tf.concat((real_labels, fake_labels), axis=0)
        combined_data = tf.concat((real_data, fake_data), axis=0)
        for _ in range(self.discriminator_epochs):
            with tf.GradientTape() as tape:
                d_loss = self.d_loss_fn(
                    combined_labels, self.discriminator(combined_data))
            grads = tape.gradient(
                d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights))
        return d_loss

    def train_generator(self, labels, data, data_type='float32'):
        """ Training step for generator."""
        g_loss = None
        for _ in range(self.generator_epochs):
            with tf.GradientTape() as tape:
                predictions = self.discriminator(self.generator(data))
                g_loss = self.g_loss_fn(labels, predictions)
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(
                grads, self.generator.trainable_weights))
        return g_loss

    def save(self, discriminator_path, generator_path):
        self.discriminator.save(discriminator_path)
        self.generator.save(generator_path)


class WGAN(GAN):
    """ Class for Wasserstein generative adversarial network."""
    def compile(
            self, d_optimizer, g_optimizer, d_loss_fn=wasserstein_loss_fn,
            g_loss_fn=wasserstein_loss_fn, **kwargs):
        """ Compile model."""
        super().compile(d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, **kwargs)

    def train_step(self, real_images):
        """ Training step iteration for model."""
        # Gather data and necessary variables
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        data_type = real_images.dtype
        batch_size = tf.shape(real_images)[0]
        shape = (batch_size, self.latent_dim)
        random_latent_vectors = tf.random.normal(shape=shape, dtype=data_type)
        # Generate fake data
        generated = self.generator(random_latent_vectors)
        # Get labels for training
        real_labels = tf.ones((batch_size, 1), dtype=data_type)
        fakes_labels = -real_labels  # -tf.ones((batch_size, 1), dtype=data_type)
        random_noise = 0  # tf.random.normal(shape=(tf.shape(real_images)), stddev=1e-12)
        # Train models
        d_loss = self.train_discriminator(
            real_labels, fakes_labels, real_images + random_noise,
            generated, data_type=data_type)
        g_loss = super().train_generator(
            real_labels, random_latent_vectors, data_type=data_type)
        # Compute metrics
        random_latent_vectors = tf.random.normal(shape=shape, dtype=data_type)
        # avg_d_loss, avg_g_loss = avg_d_loss / self.discriminator_epochs, avg_g_loss / self.generator_epochs
        self.compiled_metrics.update_state(
            real_images, self.generator(random_latent_vectors))
        metrics = {m.name: m.result() for m in self.metrics}
        wasserstein_score = wasserstein_metric_fn(
            None, self.discriminator(self.generator(random_latent_vectors)))
        my_metrics = {
            'd_loss': d_loss, 'g_loss': g_loss,
            'wasserstein_score': wasserstein_score}
        metrics.update(my_metrics)
        return metrics

    def train_discriminator(
            self, real_labels, fake_labels,
            real_data, fake_data, data_type='float32'):
        """ Training iterator for discriminator."""
        d_loss = None
        lamb = 20  # tf.constant(10, dtype=data_type)
        for _ in range(self.discriminator_epochs):
            with tf.GradientTape() as tape:
                # d_loss = self.d_loss_fn(self.discriminator(real_images), self.discriminator(generated))
                d_loss = self.d_loss_fn(
                    tf.concat((real_labels, fake_labels), axis=0),
                    tf.concat((self.discriminator(real_data),
                               self.discriminator(fake_data)), axis=0))
                val = self.gradient_penalty(gp_data=real_data)
                d_loss = d_loss + (val * lamb)
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights))
        return d_loss

    def gradient_penalty(self, gp_data):
        """ Calculates the gradient penalty.
         found in keras wgan_gp, modified
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # batch_size = tf.shape(gp_data)[0]
        # Get the interpolated image

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(gp_data)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(gp_data, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [gp_data])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp


class CWGAN(WGAN):
    """ Class for conditional WGAN."""
    def train_step(self, data):
        """ Training step iteration for model."""
        # Prepare Data
        if isinstance(data, tuple):
            data = data[0]
        real_images, class_labels = data[0], data[1]
        data_type, batch_size = real_images.dtype, tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim), dtype=data_type)
        generated = self.generator((random_latent_vectors, class_labels))
        real_labels, fakes_labels = tf.ones((batch_size, 1), dtype=data_type), -tf.ones((batch_size, 1),
                                                                                        dtype=data_type)
        # Compute Loss
        d_loss = self.train_discriminator(real_labels, fakes_labels, real_images, generated, class_labels,
                                          data_type=data_type)
        g_loss = self.train_generator(real_labels, random_latent_vectors, class_labels, data_type=data_type)
        # Get Metrics
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), dtype=data_type)
        self.compiled_metrics.update_state(real_images, self.generator((random_latent_vectors, class_labels)))
        metrics = {m.name: m.result() for m in self.metrics}
        wasserstein_score = wasserstein_metric_fn(None, self.discriminator(
            (self.generator((random_latent_vectors, class_labels)), class_labels)))
        critic_gen_loss = (-d_loss * g_loss) + g_loss
        critic_gen_diversity_loss = critic_diversity_metric(
            self.discriminator((real_images, class_labels)),
            self.discriminator((self.generator((random_latent_vectors, class_labels)), class_labels)))
        my_metrics = {
            'd_loss': d_loss, 'g_loss': g_loss, 'wasserstein_score': wasserstein_score,
            'comb_loss': critic_gen_loss, 'divergence_approx': critic_gen_diversity_loss}
        metrics.update(my_metrics)
        return metrics

    def train_discriminator(
            self, real_labels, fake_labels, real_data, fake_data,
            class_labels, data_type='float32'):
        """ Training step for discriminator."""
        lamb = 20  # tf.constant(10, dtype=data_type)
        for _ in range(self.discriminator_epochs):
            with tf.GradientTape() as tape:
                # d_loss = self.d_loss_fn(self.discriminator(real_images), self.discriminator(generated))
                d_loss = self.d_loss_fn(tf.concat((real_labels, fake_labels), 0), tf.concat(
                    (self.discriminator((real_data, class_labels)), self.discriminator((fake_data, class_labels))),
                    0))
                val = self.gradient_penalty(gp_data=real_data, labels=class_labels)
                d_loss = d_loss + (val * lamb)
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        return d_loss

    def train_generator(self, labels, data, class_labels, data_type='float32'):
        """ Training step for generator."""
        g_loss = None
        for _ in range(self.generator_epochs):
            with tf.GradientTape() as tape:
                predictions = self.discriminator(
                    (self.generator((data, class_labels)), class_labels))
                g_loss = self.g_loss_fn(labels, predictions)
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return g_loss

    def gradient_penalty(self, gp_data, labels):
        """ Calculates the gradient penalty.
         found in keras wgan_gp, modified
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # batch_size = tf.shape(gp_data)[0]
        # Get the interpolated image

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(gp_data)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator((gp_data, labels))  # , training=True

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [gp_data])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp


class Autoencoder(keras.Model):
    """ Autoencoder class."""
    def __init__(self, encoder, decoder, latent_dimension):
        super(Autoencoder, self).__init__()
        self.encoder, self.decoder, self.latent_dimension = encoder, decoder, latent_dimension

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class EBGAN(GAN):
    """ Energy Based GAN."""

    def train_step(self, real_images):
        """ Training step iteration for model."""
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        data_type, batch_size = real_images.dtype, tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim))
        g_loss = self.train_generator(
            tf.zeros((batch_size, 1)), random_latent_vectors,
            data_type=data_type)
        self.compiled_metrics.update_state(
            real_images, self.generator(random_latent_vectors))
        metrics = {m.name: m.result() for m in self.metrics}
        my_metrics = {'g_loss': g_loss, 'Reconstruction error': g_loss}
        metrics.update(my_metrics)
        return metrics

    def train_generator(self, labels, data, data_type='float32'):
        """ Train generator to reduce reconstruction error of its values
        passed through the discriminator."""
        g_loss = None
        for _ in range(self.generator_epochs):
            with tf.GradientTape() as tape:
                # reconstruction_error = self.get_reconstruction_error(self.generator(data))
                # g_loss = self.g_loss_fn(labels, reconstruction_error)
                synthetic = self.generator(data)
                g_loss = self.g_loss_fn(self.discriminator(synthetic), synthetic)
                val_loss = self.regularizer(synthetic)
                loss = g_loss + val_loss
            grads = tape.gradient(loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(
                zip(grads, self.generator.trainable_weights))
        return g_loss

    def get_reconstruction_error(self, x):
        """ Return loss function value between input and
         discriminator's reconstruction attempt."""
        return self.d_loss_fn(self.discriminator(x), x)

    def regularizer(self, data):
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(data)
            pred = self.discriminator.encoder(data)
        grads = gp_tape.gradient(pred, [data])[0]
        # batch_size = tf.cast(tf.shape(data)[0], data.dtype)
        length, sum_loss = grads.shape[0], 0.0
        sum_loss = tf.reduce_sum(tf.square(cosine_similarity(grads, grads)))
        # for a in range(length):
        #     for b in range(length):
        #         if a != b:
        #             sim = tf.square(cosine_similarity(grads[a], grads[b]))
        #             sum_loss += sim
        reg_loss = sum_loss / (length * (length - 1))
        return reg_loss


class VAE(keras.Model):
    """Convolutional variational autoencoder."""
    def __init__(self, encoder, decoder, latent_dimension):
        super(VAE, self).__init__()
        self.latent_dimension = latent_dimension
        self.encoder = encoder
        self.decoder = decoder

    def predict(self, x):
        return self.call(x)

    def call(self, inputs, training=False):
        # print(x)
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z) if training else self.sample(z)
        return x_logit

    def compile(self, optimizer, metrics=None):
        super(VAE, self).compile(optimizer, metrics=metrics)

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dimension))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        # print(x.shape)
        encoded = self.encoder(x)
        # print(encoded.shape)
        mean, logvar = tf.split(encoded, num_or_size_splits=2, axis=1)
        return mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        eps = tf.random.normal(shape=mean.shape, dtype=mean.dtype)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    @staticmethod
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
          -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar
                 + log2pi), axis=raxis)

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=x)
        print('Cross Entropy Shape: {}'.format(cross_ent.shape))
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def train_step(self, x):
        """Executes one training step and returns the loss.
        This function computes the loss and gradients,
        and uses the latter to update the model's parameters.
        """
        if isinstance(x, tuple):
            x = x[0]
        # data_type = x.dtype
        # batch_size = tf.shape(x)[0]
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.compiled_metrics.update_state(x, self.call(x))
        metrics = {m.name: m.result() for m in self.metrics}
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        metrics.update({'loss': loss})
        return metrics


# Written by fchollet 2020
# https://keras.io/examples/generative/vae/#create-a-sampling-layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
