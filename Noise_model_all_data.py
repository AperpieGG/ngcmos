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
    common_tic_ids = set(all_data[0]['TIC_IDs']).intersection(all_data[1]['TIC_IDs'])
    print(f"Found {len(common_tic_ids)} common TIC_IDs between the two JSON files")

    # Extract RMS and magnitude values for the common TIC_IDs
    common_rms = [[] for _ in range(len(all_data))]
    common_mags = [[] for _ in range(len(all_data))]
    for idx, data in enumerate(all_data):
        for tic_id, rms, mag in zip(data['TIC_IDs'], data['RMS_list'], data['mags_list']):
            if tic_id in common_tic_ids:
                common_rms[idx].append(rms)
                common_mags[idx].append(mag)
    print(f"Found {len(common_tic_ids)} common TIC_IDs between the two JSON files")
    print(f' The first tic_id has name: {all_data[0]["TIC_IDs"][0]}, and rms value for cmos is: {all_data[0]["RMS_list"][0]}, and for ccd is: {all_data[1]["RMS_list"][0]}')

    for tic_id in common_tic_ids:
        print(f"The tic_id is: {tic_id} with cmos rms value: "
              f"{all_data[0]['RMS_list'][all_data[0]['TIC_IDs'][0] == tic_id]} and ccd rms value: "
                f"{all_data[1]['RMS_list'][all_data[1]['TIC_IDs'][1] == tic_id]}")

    # Plot common RMS values against magnitude lists for both JSON files on the same plot
    plt.figure(figsize=(10, 8))
    for i in range(len(all_data)):
        file_name = json_files[i]
        label = "CMOS" if "rms_mags_phot_NG1109-2807_1.json" in file_name else "CCD"
        plt.plot(common_mags[i], common_rms[i], 'o', label=label)
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
