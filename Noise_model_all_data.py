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
    json_files = [f for f in files if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in {directory}")

    # Iterate over each JSON file
    for json_file in json_files:
        # Form the full path to the JSON file
        json_path = os.path.join(directory, json_file)

        # Load data from the JSON file
        data = load_rms_mags_data(json_path)

        # Extract information from the data
        # Example:
        RMS_list = data['RMS_list']
        mags_list = data['mags_list']

        # Do something with the extracted information
        # Example:
        print(f"RMS values: {RMS_list}")
        print(f"Magnitude values: {mags_list}")

        # Optionally, plot the data
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(mags_list, RMS_list, 'o', color='c', label='total data', alpha=0.5)

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
