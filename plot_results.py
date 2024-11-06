#!/usr/bin/env python
import argparse
from matplotlib import pyplot as plt, ticker
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
        np.where((np.array(mags_list) > 4) & (np.array(mags_list) < 9) & (np.array(RMS_list) >= 5000))[0]

    filtered_indices_dim = []
    filtered_indices_dim = np.where((np.array(mags_list) < 12) & (np.array(RMS_list) >= 20000))[0]

    return filtered_indices_bright, filtered_indices_dim


def identify_outliers(data, deviation_threshold):
    tmag_list = data['Tmag_list']
    mags_list = data['mags_list']
    tic_ids = data['TIC_IDs']
    RMS_list = data['RMS_list']  # Assuming RMS data is available in 'data'

    outliers = []

    for tmag, mag, tic_id in zip(tmag_list, mags_list, tic_ids):
        deviation = abs(tmag - mag)
        if deviation > deviation_threshold:
            # Find the index of the outlier using TIC_IDs
            outlier_index = np.where(np.array(data['TIC_IDs']) == tic_id)[0][0]
            outliers.append((tic_id, tmag, mag, RMS_list[outlier_index]))
    print("Outliers:")
    for tic_id, tmag, mag, RMS in outliers:
        print(f"TIC ID: {tic_id}, Tmag: {np.round(tmag, 2)}, Mag: {np.round(mag, 2)}, RMS: {RMS}")

    return outliers


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_noise_model(data):
    fig, ax = plt.subplots(figsize=(10, 8))
    RMS_list = data['RMS_list']
    tic_ids = data['TIC_IDs']
    color_list = data['COLOR']
    Tmag_list = data['Tmag_list']
    synthetic_mag = data['synthetic_mag']
    RNS = data['RNS']
    photon_shot_noise = data['photon_shot_noise']
    read_noise = data['read_noise']
    dc_noise = data['dc_noise']
    sky_noise = data['sky_noise']
    N = data['N']
    print(f'The average scintillation noise is: {np.mean(N)}')

    # Filter data points based on magnitude and RMS
    filtered_indices_bright, filtered_indices_dim = filter_data(Tmag_list, RMS_list)
    filtered_indices = np.append(filtered_indices_bright, filtered_indices_dim)

    # Exclude outliers from the total data
    total_RMS = [RMS_list[i] for i in range(len(RMS_list)) if i not in filtered_indices]
    total_mags = [Tmag_list[i] for i in range(len(Tmag_list)) if i not in filtered_indices]
    total_colors = [color_list[i] for i in range(len(color_list)) if i not in filtered_indices]

    # Normalize the color list to map it to the coolwarm colormap
    norm = mcolors.Normalize(vmin=min(total_colors), vmax=max(total_colors))
    cmap = cm.coolwarm

    # Scatter plot with color map
    scatter = ax.scatter(total_mags, total_RMS, c=total_colors, cmap=cmap, norm=norm, alpha=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Color Metric')  # Label for colorbar, can be adjusted

    # Plot various noise sources
    ax.plot(synthetic_mag, RNS, color='black', label='total noise')
    ax.plot(synthetic_mag, photon_shot_noise, color='green', label='photon shot', linestyle='--')
    ax.plot(synthetic_mag, read_noise, color='red', label='read noise', linestyle='--')
    ax.plot(synthetic_mag, dc_noise, color='purple', label='dark noise', linestyle='--')
    ax.plot(synthetic_mag, sky_noise, color='blue', label='sky bkg', linestyle='--')
    ax.plot(synthetic_mag, np.ones(len(synthetic_mag)) * N, color='orange', label='scintillation noise',
            linestyle='--')

    # Plot formatting
    ax.set_xlabel('TESS Magnitude')
    ax.set_ylabel('RMS (ppm)')
    ax.set_yscale('log')
    ax.set_xlim(7.5, 14)
    ax.set_ylim(1000, 100000)
    ax.invert_xaxis()
    plt.legend(loc='best')
    plt.tight_layout()

    plt.show()


def linear_model(x, m, b):
    return m * x + b


# def plot_tmag_vs_mag(data):
#     fig, ax = plt.subplots(figsize=(10, 8))
#
#     mags_list = data['mags_list']
#     print('The length of mags_list is:', len(mags_list))
#     tmag_list = data['Tmag_list']
#     print('The length of tmag_list is:', len(tmag_list))
#     RMS_list = data['RMS_list']
#
#     # Filter data points based on magnitude and RMS
#     filtered_indices_bright, filtered_indices_dim = filter_data(mags_list, RMS_list)
#
#     filtered_indices = np.append(filtered_indices_bright, filtered_indices_dim)
#
#     # # Exclude outliers from the total data
#     tmag_list = [tmag_list[i] for i in range(len(tmag_list)) if i not in filtered_indices]
#     mags_list = [mags_list[i] for i in range(len(mags_list)) if i not in filtered_indices]
#
#     ax.plot(tmag_list, mags_list, 'o', color='red', alpha=0.5)
#     ax.set_xlabel('Tmag')
#     ax.set_ylabel('Apparent Magnitude')
#     ax.set_xlim(7.5, 16)
#     ax.set_ylim(7.5, 16)
#     plt.tight_layout()
#     plt.gca().invert_xaxis()
#     plt.gca().invert_yaxis()
#     plt.show()


def main(json_file):
    # Set plot parameters
    plot_images()
    # Load RMS and magnitude data from JSON file
    data = load_rms_mags_data(json_file)

    # Plot RMS vs magnitudes
    plot_noise_model(data)
    # plot_tmag_vs_mag(data)

    # Identify outliers
    deviation_threshold = 2
    identify_outliers(data, deviation_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot RMS vs Magnitudes')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing RMS and magnitude data')
    args = parser.parse_args()

    # Run main function
    main(args.json_file)
