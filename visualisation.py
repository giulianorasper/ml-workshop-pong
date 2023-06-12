import math

import numpy as np
from matplotlib import pyplot as plt


def plot_results(values, title='Title', xlabel='xlabel', ylabel='ylabel'):

    if "loss" in title.lower():
        throw_away_amount = len(values) // 10

        # Create a new list without the first 1/10th elements
        values = values[throw_away_amount:]

    # Calculate moving averages
    window_size = len(values) // 10 + 1
    moving_averages = np.convolve(values, np.ones(window_size) / window_size, mode='valid')

    # Plot original rewards per episode
    plt.plot(values, label='Original')

    # Plot smoothed rewards per episode
    plt.plot(range(window_size - 1, len(values)), moving_averages, label='Smoothed')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()