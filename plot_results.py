#!/usr/bin/env python
import argparse
from matplotlib import pyplot as plt
import json


def load_rms_mags_data(filename):
    """
    Load RMS and magnitude data from JSON file
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def plot_rms_vs_mag(data):
    """
    Plot RMS vs magnitudes
    """
    RMS_list = data["RMS_list"]
    mag_list = data["mags_list"]

    plt.figure(figsize=(10, 6))
    plt.scatter(mag_list, RMS_list, color='blue', alpha=0.5)
    plt.xlabel('Magnitudes')
    plt.ylabel('RMS (ppm)')
    plt.title('RMS vs Magnitudes')
    plt.show()


def main(json_file):
    # Load RMS and magnitude data from JSON file
    data = load_rms_mags_data(json_file)

    # Plot RMS vs magnitudes
    plot_rms_vs_mag(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot RMS vs Magnitudes')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing RMS and magnitude data')
    args = parser.parse_args()

    # Run main function
    main(args.json_file)


