import argparse


# Need to get data, labels, time
import os
from typing import List

import numpy as np

from af_accel_GAN import standard_conditional, save_gan
from constants.af_accel_constants import CONFIG_DATA, H_PARAMS, DATA_DIR, DATA_PATH, TIME_PATH, LABEL_PATH, DATA_LENGTH, \
    DTYPE, LATENT_DIM, EPOCHS, BATCH_SIZE
from custom_functions.custom_classes import Data, Hyperparameters
from utils.toml_utils import load_toml


def data_interpolate(data: Data, hp: Hyperparameters, repeat_amounts: List, dir=None):
    results_dir = dir if dir is not None else ''
    for repeat in repeat_amounts:
        hp.num_repeats = repeat  # this is for when num_repeats gets removed
        model, _ = standard_conditional(
            data.time, data.data, data.labels, data.data_size, data.data_type,
            hp.lat_dim, hp.epochs, hp.batch_size, num_repeats=repeat)
        gen_name = f'generator-{repeat}-repeats'
        critic_name = f'critic-{repeat}_repeats'
        save_gan(model.generator, model.discriminator, results_dir, gen_name, critic_name)


def get_args():
    """ Parse command line."""
    parse = argparse.ArgumentParser()
    parse.add_argument('config_file')  # Path to config file
    args = parse.parse_args()
    config_table = load_toml(args.config_file)
    config_data = config_table[CONFIG_DATA]
    config_hp = config_table[H_PARAMS]
    data_dir = config_data[DATA_DIR]
    data = np.load(os.path.join(data_dir, config_data[DATA_PATH]))
    time = np.load(os.path.join(data_dir, config_data[TIME_PATH]))
    labels = np.load(os.path.join(data_dir, config_data[LABEL_PATH]))
    data = Data(time, data, DATA_LENGTH, config_data[DTYPE], labels)
    latent_dim = config_hp[LATENT_DIM]
    epochs = config_hp[EPOCHS]
    batch_size = config_hp[BATCH_SIZE]
    repeats = config_hp['repeats']  # This will be made into constant in future
    hp = Hyperparameters(latent_dim, epochs, batch_size)
    return data, hp, repeats


def main():
    data, hp, repeats = get_args()
    data_interpolate(data, hp, repeats, dir=os.path.join('results', 'data_interpolation'))


if __name__ == '__main__':
    main()
