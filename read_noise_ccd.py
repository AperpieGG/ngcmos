#! /usr/bin/env python
import argparse
import glob
import os
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from utils import plot_images


def read_bias_data():
    path = os.getcwd()

    list_images = glob.glob(path + '/*.fits.bz2')
    if len(list_images) == 0:
        list_images = glob.glob(path + '/*.fits')
    bias_values = []

    for image_path in list_images:
        hdulist = fits.open(image_path)
        image_data = hdulist[0].data[0:2048, 0:2048]
        hdulist.close()
        bias_values.append(image_data)

    bias_values = np.array(bias_values)
    print('The shape of the bias_values is ', bias_values.shape)

    value_mean = np.mean(bias_values, axis=0).flatten()
    value_std = np.std(bias_values, axis=0).flatten()
    print('The value_mean and value_std are [{}] and [{}]'.format(value_mean, value_std))
    return value_mean, value_std


def plot_read_noise(value_mean, value_std):
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0, hspace=0)

    ax1 = plt.subplot(gs[0])
    print(f'Type of value_mean is {type(value_mean)}, shape is {value_mean.shape}')
    print(f'Type of value_std is {type(value_std)}, shape is {value_std.shape}')
    hb = ax1.hist2d(value_mean, value_std, bins=100, cmap='cividis', norm=LogNorm())
    ax1.set_xlabel('Mean (ADU)')
    ax1.set_ylabel('RMS (ADU)')

    ax2 = plt.subplot(gs[1])
    ax2.hist(value_std, bins=100, orientation='horizontal', color='b', histtype='step')
    ax2.set_xlabel('Number of pixels')
    ax2.set_xticklabels([], minor=True)
    ax2.set_xscale('log')

    value_median_hist = np.median(value_std)
    print('Value Median = ', value_median_hist)
    value_mean_hist = np.mean(value_std)
    print('Value Mean = ', value_mean_hist)
    RMS = np.sqrt(np.mean(value_std ** 2))
    print('RMS = ', RMS)
    ax2.axhline(value_median_hist, color='g', linestyle=':')
    ax2.yaxis.set_ticklabels([])

    y_min, y_max = ax1.get_ylim()
    ax2.set_ylim(y_min, y_max)

    fig.colorbar(ax1.get_children()[0], ax=ax2, label='Number of pixels')
    fig.tight_layout()

    plt.show()


# value_mean, value_std = calculate_read_noise()
# plot_read_noise(value_mean, value_std)

def main():
    plot_images()
    value_mean, value_std = read_bias_data()
    plot_read_noise(value_mean, value_std)


if __name__ == "__main__":
    main()
