#!/usr/bin/env python
"""
Plot the noise model between same tic_ids of CCD and CMOS, expecting same NG field and same night
"""
import argparse
from matplotlib import pyplot as plt
import json
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


def process_json_files(directory, field):
    """
    Process JSON files in the specified directory
    """
    # Get a list of all files in the directory
    files = os.listdir(directory)
    # Filter out only the JSON files
    cmos_json_file = [f for f in files if f.startswith(f'rms_mags_phot_{field}_1') and f.endswith('.json')]
    ccd_json_file = [f for f in files if f.startswith(f'rms_mags_phot_{field}_ccd_1') and f.endswith('.json')]
    json_files = cmos_json_file + ccd_json_file

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
    common_tmag = set(all_data[1]['TIC_IDs']).intersection(all_data[0]['TIC_IDs'])
    print(f"Found {len(common_tmag)} common Tmags between the two JSON files")
    # Extract RMS and magnitude values for the common TIC_IDs
    # Extract RMS and magnitude values for the common Tmag values
    common_rms = [[] for _ in range(len(all_data))]
    common_mags = [[] for _ in range(len(all_data))]
    for idx, data in enumerate(all_data):
        for tmag, rms in zip(data['mags_list'], data['RMS_list']):
            if tmag in common_tmag:
                common_rms[idx].append(rms)
                common_mags[idx].append(tmag)

    # Plot common RMS values against magnitude lists for both JSON files on the same plot
    plt.figure(figsize=(10, 8))
    for i in range(len(all_data)):
        file_name = json_files[i]
        label = "CMOS" if "rms_mags_phot_NG1109-2807_1.json" in file_name else "CCD"
        plt.plot(common_mags[i], common_rms[i], 'o', label=label, alpha=0.8)

        # # Plot lines connecting corresponding data points with the same tmag
        # for tmag, rms in zip(common_mags[i], common_rms[i]):
        #     plt.plot([tmag, tmag], [0, rms], 'k--', alpha=0.5)

        for tmag in common_tmag:
            idx1 = common_mags[0].index(tmag)
            idx2 = common_mags[1].index(tmag)
            plt.plot([common_mags[0][idx1], common_mags[1][idx2]], [common_rms[0][idx1], common_rms[1][idx2]], 'k-',
                     alpha=0.2)

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
    # Set plot parameters
    plot_images()
    # Get the current working directory
    directory = os.getcwd()
    # add the field name by argument

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot noise model for CCD and CMOS data')
    parser.add_argument('field', type=str, help='The NG field name')
    args = parser.parse_args()
    field = args.field

    # Process JSON files in the specified directory
    process_json_files(directory, field)


if __name__ == "__main__":
    main()
