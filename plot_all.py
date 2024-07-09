#!/usr/bin/env python
import argparse
from matplotlib import pyplot as plt, ticker
import json
import numpy as np
import os
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

    ax.plot(mags_list, RMS_list, 'o', color='c', label='total data', alpha=0.5)
    # ax.plot(synthetic_mag, RNS, color='c', label='total noise')
    # ax.plot(synthetic_mag, photon_shot_noise, color='green', label='photon shot', linestyle='--')
    # ax.plot(synthetic_mag, read_noise, color='red', label='read noise', linestyle='--')
    # ax.plot(synthetic_mag, dc_noise, color='purple', label='dark noise', linestyle='--')
    # ax.plot(synthetic_mag, sky_noise, color='blue', label='sky bkg', linestyle='--')
    # ax.plot(synthetic_mag, np.ones(len(synthetic_mag)) * N, color='orange', label='scintillation noise',
    #         linestyle='--')
    ax.set_xlabel('TESS Magnitude')
    ax.set_ylabel('RMS (ppm)')
    ax.set_yscale('log')
    ax.set_xlim(7.5, 14)
    ax.set_ylim(1000, 100000)
    ax.invert_xaxis()
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_tmag_vs_mag(data):
    fig, ax = plt.subplots(figsize=(10, 8))

    mags_list = data['mags_list']
    tmag_list = data['Tmag_list']

    ax.plot(tmag_list, mags_list, 'o', color='red', alpha=0.5)
    ax.set_xlabel('Tmag')
    ax.set_ylabel('Apparent Magnitude')
    ax.set_xlim(7.5, 16)
    ax.set_ylim(7.5, 16)
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()


def main(directory):
    # Set plot parameters
    plot_images()

    # Iterate over all JSON files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            json_file = os.path.join(directory, filename)
            print(f"Processing file: {json_file}")

            # Load RMS and magnitude data from JSON file
            data = load_rms_mags_data(json_file)
            for i in data:
                plot_noise_model(data[i])
                plot_tmag_vs_mag(data[i])


if __name__ == "__main__":
    directory = '.'
    main(directory)
