#!/usr/bin/env python
from matplotlib import pyplot as plt
import json
import numpy as np
from utils import plot_images


# dark background
# plt.style.use('dark_background')


def load_rms_mags_data(filename):
    """
    Load RMS and magnitude data from JSON file
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def filter_data(mags_list, RMS_list):
    """
    Filter data points based on magnitude and RMS criteria
    """
    filtered_indices_bright = []
    filtered_indices_bright = \
        np.where((np.array(mags_list) > 4) & (np.array(mags_list) < 9.5) & (np.array(RMS_list) >= 5000))[0]

    filtered_indices_dim = []
    filtered_indices_dim = np.where((np.array(mags_list) < 12) & (np.array(RMS_list) >= 20000))[0]

    return filtered_indices_bright, filtered_indices_dim


def plot_noise_model(data_bright, data_dark):
    fig, ax = plt.subplots(figsize=(10, 8))
    RMS_list_bright = data_bright['RMS_list']
    mags_list_bright = data_bright['mags_list']
    RMS_list_dark = data_dark['RMS_list']
    mags_list_dark = data_dark['mags_list']

    # Filter data points based on magnitude and RMS
    filtered_indices_bright_bright, filtered_indices_dim_bright = filter_data(mags_list_bright, RMS_list_bright)
    filtered_indices_bright_dark, filtered_indices_dim_dark = filter_data(mags_list_dark, RMS_list_dark)

    # append the indices of the outliers
    filtered_indices_bright = np.append(filtered_indices_bright_bright, filtered_indices_dim_bright)
    filtered_indices_dark = np.append(filtered_indices_bright_dark, filtered_indices_dim_dark)

    # # Exclude outliers from the total data
    total_RMS_list_bright = [RMS_list_bright[i] for i in range(len(RMS_list_bright)) if i not in filtered_indices_bright]
    total_mags_list_bright = [mags_list_bright[i] for i in range(len(mags_list_bright)) if i not in filtered_indices_bright]
    total_RMS_list_dark = [RMS_list_dark[i] for i in range(len(RMS_list_dark)) if i not in filtered_indices_dark]
    total_mags_list_dark = [mags_list_dark[i] for i in range(len(mags_list_dark)) if i not in filtered_indices_dark]

    ax.plot(total_mags_list_bright, total_RMS_list_bright, 'o', color='r', label='bright data', alpha=0.5)
    ax.plot(total_mags_list_dark, total_RMS_list_dark, 'o', color='b', label='dark data', alpha=0.5)
    ax.set_xlabel('TESS Magnitude')
    ax.set_ylabel('RMS (ppm)')
    ax.set_yscale('log')
    ax.set_xlim(7.5, 14)
    ax.set_ylim(1000, 100000)
    ax.invert_xaxis()
    plt.legend(loc='best')
    plt.tight_layout()

    plt.show()


def main():
    # Set plot parameters
    plot_images()
    # Load RMS and magnitude data from JSON file
    data_bright = load_rms_mags_data('rms_mags_rel_phot_NG2320-1302_1_0622.json')
    data_dark = load_rms_mags_data('rms_mags_rel_phot_NG2320-1302_1_0705.json')

    # Plot RMS vs magnitudes
    plot_noise_model(data_bright, data_dark)


if __name__ == "__main__":
    # Run main function
    main()
