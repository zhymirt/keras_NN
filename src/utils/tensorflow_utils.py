import tensorflow as tf


def data_to_dataset(data, dtype='float32', batch_size=64, shuffle=True):
    """ Return dataset given numpy data."""
    dataset = tf.data.Dataset.from_tensor_slices(tf.cast(data, dtype=dtype))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset
