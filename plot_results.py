#!/usr/bin/env python
import argparse
from matplotlib import pyplot as plt
import json
import numpy as np
from utils import plot_images

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
    filtered_indices = np.where((np.array(mags_list) > 7.5) & (np.array(mags_list) < 10) & (np.array(RMS_list) >= 6000))[0]
    return filtered_indices


def plot_noise_model(data):
    fig, ax = plt.subplots(figsize=(10, 8))
    RMS_list = data['RMS_list']
    mags_list = data['mags_list']
    synthetic_mag = data['synthetic_mag']
    RNS = data['RNS']
    photon_shot_noise = data['photon_shot_noise']
    read_noise = data['read_noise']
    dc_noise = data['dc_noise']
    sky_noise = data['sky_noise']
    N = data['N']

    # Filter data points based on magnitude and RMS
    filtered_indices = filter_data(mags_list, RMS_list)

    # Plot total data excluding filtered points
    total_indices = [i for i in range(len(mags_list)) if i not in filtered_indices]
    total_mags = [mags_list[i] for i in total_indices]
    total_RMS = [RMS_list[i] for i in total_indices]

    ax.plot(total_mags, total_RMS, 'o', color='darkgreen', label='total data', alpha=0.5)

    ax.plot(synthetic_mag, RNS, color='black', label='total noise')
    ax.plot(synthetic_mag, photon_shot_noise, color='green', label='photon shot', linestyle='--')
    ax.plot(synthetic_mag, read_noise, color='red', label='read noise', linestyle='--')
    ax.plot(synthetic_mag, dc_noise, color='purple', label='dark noise', linestyle='--')
    ax.plot(synthetic_mag, sky_noise, color='blue', label='sky bkg', linestyle='--')
    ax.plot(synthetic_mag, np.ones(len(synthetic_mag)) * N, color='orange', label='scintillation noise',
            linestyle='--')
    ax.set_xlabel('TESS Magnitude')
    ax.set_ylabel('RMS (ppm)')
    ax.set_yscale('log')
    ax.set_xlim(7.5, 14)
    ax.set_ylim(1000, 100000)
    ax.invert_xaxis()
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_tmag_vs_mag(tmag_list, mags_list):
    fig, ax = plt.subplots(figsize=(10, 8))

    filtered_indices = filter_data(mags_list, tmag_list)
    total_indices = [i for i in range(len(mags_list)) if i not in filtered_indices]
    filtered_mags = [mags_list[i] for i in total_indices]
    filtered_tmags = [tmag_list[i] for i in total_indices]

    ax.plot(filtered_tmags, filtered_mags, 'o', color='darkgreen', label='data', alpha=0.5)
    ax.set_xlabel('Tmag')
    ax.set_ylabel('Mean Magnitude')
    ax.set_xlim(7.5, 16)
    ax.set_ylim(7.5, 16)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()


def main(json_file):
    # Set plot parameters
    plot_images()
    # Load RMS and magnitude data from JSON file
    data = load_rms_mags_data(json_file)

    # Plot RMS vs magnitudes
    plot_noise_model(data)
    plot_tmag_vs_mag(data['Tmag_list'], data['mags_list'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot RMS vs Magnitudes')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing RMS and magnitude data')
    args = parser.parse_args()

    # Run main function
    main(args.json_file)


