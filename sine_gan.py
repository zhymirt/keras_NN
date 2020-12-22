import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers

if __name__ == '__main__':
    latent_dimension = 200
    vector_size = 100
    # create discriminator and generator
    # discrim = keras.Sequential(
    #     [
    #         Dense(10, input_shape=(vector_size,), activation='relu'),
    #         Dense(100, activation='relu'),
    #         Dense(1, activation='sigmoid')
    #     ]
    # )
    # discrim.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True))
    # generator = keras.Sequential([
    #     Dense(100, activation='relu', input_shape=(latent_dimension,)),
    #     Dense(100, activation='relu'),
    #     Dense(100, activation='relu'),
    #     Dense(40, activation='relu'),
    #     Dense(vector_size, activation='relu')
    # ])
    discrim = keras.Sequential(
    [
        layers.Reshape((vector_size, 1,), input_shape=(vector_size,)),
        layers.Conv1D(64, (3), strides=(2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(128, (3), strides=(2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling1D(),
        layers.Reshape((128,)),
        layers.Dense(1, activation='sigmoid'),
    ],
    name="discriminator",
    )
    # print(generator.input_shape)
    # print(generator.output_shape)
    generator = keras.Sequential(
        [
            layers.Reshape((latent_dimension, 1,), input_shape=(latent_dimension,)),
            layers.Conv1DTranspose(64, (3), strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(128, (3), strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1D(1, (4), strides=2, padding='same', activation='sigmoid'),
            layers.Reshape((400,)),
            layers.Dense(300),
            layers.Dense(200, activation=tf.math.cos),
            layers.Dense(vector_size, activation='tanh')
        ],
        name="generator",
    )
    # generator attempts to produce even numbers, discriminator will tell if true or not
    data_type = 'float32'
    range_min, random_range, data_size, batch_size = 0, 50, 1e4, 1
    even_min, even_range = range_min, int(random_range / 2)
    trained, passes, min_passes = False, 0, 3
    label_alias = {'fake': 0, 'real': 1}
    starts = [randint(0, 200)/100 for _ in range(int(data_size))] # generate n beginning data points
    benign_data = [generate_sine(val, val + 2, 100, frequency=randint(1, 3)) for val in starts] # generate 100 points of sine wave
    for idx in range(4):
        plot_data(benign_data[idx], show=True)
    # benign_data = [[math.sin(val)] for val in range(data_size)] # for sine numbers
    # print('Benign data: {}'.format(benign_data))
    dataset = tf.data.Dataset.from_tensor_slices(tf.cast(benign_data, dtype=data_type))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    # print('Dataset: {}'.format(dataset.take(10)))
    gan = GAN(discriminator=discrim, generator=generator, latent_dim=latent_dimension)
    gan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                # g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                # d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
                g_loss_fn=GeneratorWassersteinLoss(),
                d_loss_fn=DiscriminatorWassersteinLoss()
    )
    gan.fit(dataset.take(6), epochs=50)
    # for num in range(-20, 21):
    #     random_latent_vectors = tf.random.normal(shape=(1, latent_dimension))
        # print('Value at {}: {}'.format(num, generator.predict(random_latent_vectors)))
    # for layer in generator.layers:
    #     print(layer.get_weights())
    plot_data(generator.predict(tf.zeros(shape=(1, latent_dimension)))[0], show=True)
    for _ in range(3):
        plot_data(generator.predict(tf.random.normal(shape=(1, latent_dimension)))[0], show=True)
    plot_data(generator.predict(tf.ones(shape=(1, latent_dimension)))[0], show=True)
