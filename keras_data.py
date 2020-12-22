import numpy as np
from matplotlib import pyplot as plt


def generate_sine(start, end, points, amplitude=1, frequency=1):
    time = np.linspace(0, 2, 100)
    signal = amplitude*np.sin(2*np.pi*frequency*time)
    return signal

def plot_data(data, show=False):
    plt.plot(np.linspace(0, 2, num=100), data)
    if show:
        plt.show()
