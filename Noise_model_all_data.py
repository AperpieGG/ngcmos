#!/usr/bin/env python
import argparse
from matplotlib import pyplot as plt, ticker
import json
import numpy as np
from utils import plot_images
import os

# dark background
# plt.style.use('dark_background')


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
    json_files = [f for f in files if f.startswith('rms_mags_phot_NG1109') and
                  f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in {directory}")

    # Lists to store data from all JSON files
    all_RMS_lists = []
    all_mags_lists = []
    common_indices = None  # Initialize common_indices outside the loop

    # Iterate over each JSON file
    for json_file in json_files:
        # Form the full path to the JSON file
        json_path = os.path.join(directory, json_file)

        # Load data from the JSON file
        data = load_rms_mags_data(json_path)

        # Extract information from the data
        RMS_list = data['RMS_list']
        mags_list = data['mags_list']
        TIC_IDs = data['TIC_IDs']

        # check which TIC_IDs are common to all JSON files
        if common_indices is None:
            common_indices = set(TIC_IDs)
        else:
            common_indices = common_indices.intersection(TIC_IDs)
            print(f"The common TIC_IDs are: {common_indices} and the number of common TIC_IDs is {len(common_indices)}")

        # Append the data to the lists
        all_RMS_lists.append(RMS_list)
        all_mags_lists.append(mags_list)

    common_indices = list(common_indices)

    if len(common_indices) == 0:
        print("No common TIC_IDs found in the JSON files")
        return

    common_RMS_lists = []
    common_mags_lists = []

    for i, json_file in enumerate(json_files):
        # Extract information from the data
        RMS_list = all_RMS_lists[i]
        mags_list = all_mags_lists[i]
        TIC_IDs = data['TIC_IDs']

        # Find the common indices
        common_indices = [i for i, tic_id in enumerate(TIC_IDs) if tic_id in common_indices]

        # Ensure common indices exist before appending data
        if common_indices:
            # Append the common data to the lists
            common_RMS_lists.append([RMS_list[i] for i in common_indices])
            common_mags_lists.append([mags_list[i] for i in common_indices])
            print(f'The common_rms_list is {common_RMS_lists} and the common_mags_list is {common_mags_lists}')
        else:
            print("No common TIC_IDs found in this file:", json_file)

    # Plot all data on the same figure
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, json_file in enumerate(json_files):
        if "rms_mags_phot_NG1109-2807_ccd_1.json" in json_file:
            label = "RMS CCD"
        elif "rms_mags_phot_NG1109-2807_1.json" in json_file:
            label = "RMS CMOS"
        else:
            label = json_file
        ax.plot(common_mags_lists, common_RMS_lists, 'o', label=label)
    ax.set_xlabel('TESS Magnitude')
    ax.set_ylabel('RMS (ppm)')
    ax.set_yscale('log')
    ax.set_xlim(7.5, 14)
    ax.set_ylim(1000, 100000)
    ax.invert_xaxis()
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.tight_layout()
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
