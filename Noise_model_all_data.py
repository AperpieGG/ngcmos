#!/usr/bin/env python
import argparse
from matplotlib import pyplot as plt, ticker
import json
import numpy as np
from utils import plot_images
import os

# dark background
plt.style.use('dark_background')


def load_rms_mags_data(filename):
    """
    Load RMS and magnitude data from JSON file
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def process_json_files(directory):
    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Filter out only the JSON files
    json_files = [f for f in files if f.startswith('rms_mags_phot_NG1109') and f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in {directory}")

    # Lists to store TIC_IDs from all JSON files
    all_TIC_IDs = []

    # Iterate over each JSON file
    for json_file in json_files:
        # Form the full path to the JSON file
        json_path = os.path.join(directory, json_file)

        # Load data from the JSON file
        data = load_rms_mags_data(json_path)

        # Extract TIC_IDs from the data
        TIC_IDs = data['TIC_IDs']

        # Append the TIC_IDs to the list
        all_TIC_IDs.append(set(TIC_IDs))

    # Find the common TIC_IDs between all JSON files
    common_TIC_IDs = set.intersection(*all_TIC_IDs)

    # Print the number of common TIC_IDs
    print(f"Number of common TIC_IDs: {len(common_TIC_IDs)}")

    # Plot all data on the same figure for stars with common TIC_IDs
    fig, ax = plt.subplots(figsize=(10, 8))
    for json_file in json_files:
        # Form the full path to the JSON file
        json_path = os.path.join(directory, json_file)

        # Load data from the JSON file
        data = load_rms_mags_data(json_path)

        # Extract information from the data
        TIC_IDs = data['TIC_IDs']
        RMS_list = data['RMS_list']
        mags_list = data['mags_list']

        print(f"Common indices for {json_file}: {common_indices}")
        print(f"Length of RMS_list: {len(RMS_list)}")
        print(f"Length of mags_list: {len(mags_list)}")
        print(f"Length of TIC_IDs: {len(TIC_IDs)}")

        # Filter the data for common TIC_IDs
        common_indices = [i for i, tic_id in enumerate(TIC_IDs) if tic_id in common_TIC_IDs]
        common_RMS_list = [RMS_list[i] for i in common_indices]
        common_mags_list = [mags_list[i] for i in common_indices]

        # Plot the filtered data
        ax.plot(common_mags_list, common_RMS_list, 'o', label=json_file)

    ax.set_xlabel('TESS Magnitude')
    ax.set_ylabel('RMS (ppm)')
    ax.set_yscale('log')
    ax.set_xlim(7.5, 14)
    ax.set_ylim(1000, 100000)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.legend()
    plt.show()


def main():
    plot_images()
    parser = argparse.ArgumentParser(description="Process JSON files")
    parser.add_argument("directory", help="Directory containing JSON files")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print("Error: Directory not found, will use the current directory")
        args.directory = os.getcwd()
        return
    # Process JSON files in the specified directory
    process_json_files(args.directory)


if __name__ == "__main__":
    main()
