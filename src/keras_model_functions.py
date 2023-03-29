#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 23:26:24 2020

@author: downey
"""


import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
# from pyts.datasets import load_gunpoint


def plot_recurrence(
        y_values, rp, single_sample=True, show=False, save=False,
        save_name=None):
    x_rp = rp.fit_transform(
        y_values.reshape(1, -1) if single_sample else y_values.reshape(-1, 1))
    x_rec = x_rp[0]
    if show:
        # Show the results for the first time series
        plt.figure(figsize=(5, 5))
        plt.imshow(x_rec, cmap='binary', origin='lower')
        plt.title('Recurrence Plot', fontsize=16)
        plt.tight_layout()
        if save and save_name is not None and save_name != '':
            plt.savefig(save_name)
        if show:
            plt.show()


def get_recurrence(y_values, rp, single_sample=True):
    x_rp = rp.fit_transform(
        y_values.reshape(1, -1) if single_sample else y_values.reshape(-1, 1))
    x_rec = x_rp[0]
    return x_rec


def get_recurrence_diff(y_1, y_2):
    return y_1 - y_2


def main():
    plt.close('all')

    # X, _, _, _ = load_gunpoint(return_X_y=True)

    xx = np.linspace(0, 2, 1000)
    yy = np.sin(xx)

    yyy = yy.reshape(1, -1)
    # (-1, 1) if single feature (1, -1) if single sample
    # Recurrence plot transformation
    rp = RecurrencePlot(threshold='point', percentage=20)
    x_rp = rp.fit_transform(yyy)
    xx_image = x_rp[0]
    # Show the results for the first time series
    plt.figure(figsize=(5, 5))
    plt.imshow(xx_image, cmap='binary', origin='lower')
    plt.title('Recurrence Plot', fontsize=16)
    plt.tight_layout()
    plt.show()

    plot_recurrence(yy, rp, single_sample=True, show=True, save=False)


if __name__ == '__main__':
    main()
