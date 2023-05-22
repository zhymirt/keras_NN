import argparse


# Need to get data, labels, time
import os
from typing import List

import numpy as np
import tomlkit

from af_accel_GAN import standard_conditional, save_gan
from constants.af_accel_constants import CONFIG_DATA, H_PARAMS, DATA_DIR, DATA_PATH, TIME_PATH, LABEL_PATH, DATA_LENGTH, \
    DTYPE, LATENT_DIM, EPOCHS, BATCH_SIZE
from constants.experiments_constants import EXPERIMENTS, REPEATS, RES_DIR, BATCHES
from custom_functions.custom_classes import Data, Hyperparameters
from utils.toml_utils import load_toml


def data_interpolate(data: Data, hp: Hyperparameters, repeat_amounts: List, folder=None):
    results_dir = folder if folder is not None else ''
    for repeat in repeat_amounts:
        hp.num_repeats = repeat  # this is for when num_repeats gets removed
        model, _ = standard_conditional(
            data.time, data.data, data.labels, data.data_size, data.data_type,
            hp.lat_dim, hp.epochs, hp.batch_size, num_repeats=repeat)
        gen_name = f'generator-{repeat}-repeats'
        critic_name = f'critic-{repeat}-repeats'
        save_gan(model.generator, model.discriminator, results_dir, gen_name, critic_name)
        del model


def batch_interpolate(data: Data, hp: Hyperparameters, batch_amounts: List, folder=None):
    results_dir = folder if folder is not None else ''
    for batch in batch_amounts:
        hp.batch_size = batch  # this is for when num_repeats gets removed
        model, _ = standard_conditional(
            data.time, data.data, data.labels, data.data_size, data.data_type,
            hp.lat_dim, hp.epochs, batch, num_repeats=hp.num_repeats)
        gen_name = f'generator-{batch}-batch_size'
        critic_name = f'critic-{batch}-batch_size'
        save_gan(model.generator, model.discriminator, results_dir, gen_name, critic_name)
        del model


def check_dir_exists(config, key):
    return config[key] if key in config else ''


def get_data_from_config(config_file: tomlkit.TOMLDocument) -> Data:
    """ Parse through data section of config file and return Data object."""
    config_data = config_file[CONFIG_DATA]
    data_dir = check_dir_exists(config_data, DATA_DIR)
    data = np.load(os.path.join(data_dir, config_data[DATA_PATH]))
    time = np.load(os.path.join(data_dir, config_data[TIME_PATH]))
    labels = np.load(os.path.join(data_dir, config_data[LABEL_PATH]))
    data = Data(time, data, DATA_LENGTH, config_data[DTYPE], labels)
    return data


def get_args():
    """ Parse command line."""
    parse = argparse.ArgumentParser()
    parse.add_argument('config_file')  # Path to config file
    subparsers = parse.add_subparsers()
    parser_di = subparsers.add_parser('data-interpolation', aliases=['di'])
    parser_di.set_defaults(func=di_parser)
    parser_bi = subparsers.add_parser('batch-interpolation', aliases=['bi'])
    parser_bi.set_defaults(func=bi_parser)
    # Parse arguments
    args = parse.parse_args()
    args.func(args)


def bi_parser(args):
    config_table = load_toml(args.config_file)
    config_hp = config_table[H_PARAMS]
    data = get_data_from_config(config_table)
    latent_dim = config_hp[LATENT_DIM]
    epochs = config_hp[EPOCHS]
    batches = config_table[EXPERIMENTS][BATCHES]
    results_dir = config_table[EXPERIMENTS][RES_DIR]
    hp = Hyperparameters(latent_dim, epochs, batches[0])
    batch_interpolate(data, hp, batches, folder=os.path.join(results_dir, 'batch_interpolation'))
    return data, hp, batches


def di_parser(args):
    config_table = load_toml(args.config_file)
    config_hp = config_table[H_PARAMS]
    data = get_data_from_config(config_table)
    latent_dim = config_hp[LATENT_DIM]
    epochs = config_hp[EPOCHS]
    batch_size = config_hp[BATCH_SIZE]
    repeats = config_table[EXPERIMENTS][REPEATS]
    results_dir = config_table[EXPERIMENTS][RES_DIR]
    hp = Hyperparameters(latent_dim, epochs, batch_size)
    data_interpolate(data, hp, repeats, folder=os.path.join(results_dir, 'data_interpolation'))
    return data, hp, repeats


def main():
    get_args()
    # data, hp, repeats = get_args()
    # data_interpolate(data, hp, repeats, folder=os.path.join('results', 'data_interpolation'))


if __name__ == '__main__':
    main()
