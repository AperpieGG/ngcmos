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
    all_data = []
    # Iterate over each JSON file
    for json_file in json_files:
        # Form the full path to the JSON file
        json_path = os.path.join(directory, json_file)
        # Load data from the JSON file
        data = load_rms_mags_data(json_path)
        all_data.append(data)

    # Find the common TIC_IDs between the two JSON files
    common_tmag = set(all_data[0]['Tmag_list']).intersection(all_data[1]['Tmag_list'])
    common_tic_ids = set(all_data[0]['TIC_IDs']).intersection(all_data[1]['TIC_IDs'])
    print(f"Found {len(common_tmag)} common Tmag values between the two JSON files")
    print(f"Found {len(common_tic_ids)} common TIC_IDs between the two JSON files")
    common_tic_tmag = [(tic_id, tmag) for tic_id, tmag in zip(all_data[0]['TIC_IDs'], all_data[0]['Tmag_list']) if
                       tic_id in common_tic_ids]

    print(f"Found {len(common_tic_tmag)} common TIC_IDs and Tmag values between the two JSON files")

    plt.figure(figsize=(10, 8))

    # Plot for each JSON file
    for idx, data in enumerate(all_data):
        file_name = json_files[idx]
        label = "CMOS" if "rms_mags_phot_NG1109-2807_1.json" in file_name else "CCD"

        for tic_id, tmag in common_tic_tmag:
            # Find the index of the TIC ID in the data
            idx = data['TIC_IDs'].index(tic_id)
            # Extract RMS and magnitude values
            rms = data['RMS_list'][idx]
            mag = data['mags_list'][idx]
            plt.plot(mag, rms, 'o', label=label)

    plt.xlabel('TESS Magnitude')
    plt.ylabel('RMS (ppm)')
    plt.yscale('log')
    plt.xlim(7.5, 14)
    plt.ylim(1000, 100000)
    plt.gca().invert_xaxis()
    plt.legend()
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
