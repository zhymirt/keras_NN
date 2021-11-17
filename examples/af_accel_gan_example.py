import argparse
import sys
import tensorflow as tf
import numpy as np
from numpy import ndarray
from tensorflow.keras.models import load_model, Model


def load_saved_model(filepath: str) -> Model:
    """ Load model from given filepath."""
    generator = load_model(filepath)
    return generator


def load_data(filepath: str) -> ndarray:
    """ Load data from given filepath.
        Only accepts .npy or .npz files as input"""
    data = np.load(filepath)
    return data


def make_predictions(
        model: Model, input_data: ndarray,
        output_path: str=None) -> ndarray:
    """ Return array of predictions."""
    predictions = model.predict(input_data)
    if output_path:
        np.save(output_path, predictions)
    return predictions


def make_noise(input_shape: tuple):
    """ Return normally distributed floating point values as
        array of given shape."""
    noise = tf.random.normal(input_shape)
    return noise


# def make_class_labels(, number_of_labels: int):
#     pass


def easy_function():
    """ Function to easily generate data."""
    model_path = './conditional_af_accel_generator_v3'
    prediction_output_path = './prediction_output.npy'
    model = load_saved_model(model_path)
    input_noise = make_noise((4, 128))
    input_labels = np.array((
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]))
    predictions = make_predictions(
        model, (input_noise, input_labels), prediction_output_path)


def main():
    if len(sys.argv) <= 1:
        easy_function()
    else:
        parser = argparse.ArgumentParser(description='Generate signals.')
        parser.add_argument('model_path')
        parser.add_argument('input_files', nargs=2)
        parser.add_argument('--output')
        args = parser.parse_args()
        model = load_saved_model(args.model_path)
        noise, labels = args.input_files
        noise, labels = load_data(noise), load_data(labels)
        output_path = args.output
        assert noise.shape[0] == labels.shape[0]
        make_predictions(
            model, (noise, labels), output_path if output_path else None)


if __name__ == '__main__':
    main()
