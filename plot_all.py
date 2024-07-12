#!/usr/bin/env python
import argparse
from matplotlib import pyplot as plt
import json
import numpy as np
import os
from utils import plot_images


def load_rms_mags_data(filename):
    """
    Load RMS and magnitude data from JSON file
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def plot_noise_model(all_data):
    fig, ax = plt.subplots(figsize=(10, 8))

    for data in all_data:
        RMS_list = data['RMS_list']
        mags_list = data['mags_list']
        ax.plot(mags_list, RMS_list, 'o', alpha=0.5, color='black')

    ax.set_xlabel('TESS Magnitude')
    ax.set_ylabel('RMS (ppm)')
    ax.set_yscale('log')
    ax.set_xlim(9, 14)
    ax.set_ylim(1000, 100000)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.show()


def plot_tmag_vs_mag(all_data):
    fig, ax = plt.subplots(figsize=(10, 8))

    for data in all_data:
        mags_list = data['mags_list']
        tmag_list = data['Tmag_list']
        ax.plot(tmag_list, mags_list, 'o', alpha=0.5)

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

    all_data = []

    # Iterate over all JSON files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            json_file = os.path.join(directory, filename)
            print(f"Processing file: {json_file}")

            # Load RMS and magnitude data from JSON file
            data = load_rms_mags_data(json_file)
            all_data.append(data)

            # Print the value of N for the current file
            print(f"N for {filename}: {data['N']}")

    # Plot combined results
    plot_noise_model(all_data)
    plot_tmag_vs_mag(all_data)


if __name__ == "__main__":
    directory = '.'
    # Run main function
    main(directory)