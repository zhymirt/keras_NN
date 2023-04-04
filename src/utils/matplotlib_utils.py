import os
import matplotlib.pyplot as plt


def plot_data(
        x_values, y_values, trend_data=None, show=False,
        save=True, save_path=''):
    """ Plot (x, y) pairs, plot secondary trendline if provided."""
    plt.plot(x_values, y_values)
    if trend_data is not None and len(trend_data) > 0:  # Might be better represented as if trend_data
        plt.plot(x_values, trend_data)
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def save_wrapper(fig, save=False, save_folder=None, save_path=None):
    folder = save_folder if save_folder is not None else os.curdir
    plt.figure(fig)
    if save and save_path is not None:
        plt.savefig(os.path.join(folder, save_path))
