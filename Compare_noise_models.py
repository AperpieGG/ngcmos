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

    # Find the common tmags between the two JSON files
    common_tmag = set(all_data[0]['Tmag_list']).intersection(all_data[1]['Tmag_list'])
    print(f"Found {len(common_tmag)} common Tmag values between the two JSON files")

    # Dictionary to store additional rms and mag values for each common tmag
    additional_data = {tmag: {'rms': [], 'mag': []} for tmag in common_tmag}
    print(f"Additional data dictionary: {additional_data}")

    # Iterate over each JSON file
    for data in all_data:
        # Iterate over each entry
        for tmag, rms, mag in zip(data['Tmag_list'], data['RMS_list'], data['mags_list']):
            # Check if tmag is in common tmags
            if tmag in common_tmag:
                # Append rms and mag values to the dictionary
                additional_data[tmag]['rms'].append(rms)
                additional_data[tmag]['mag'].append(mag)

    # Plot common RMS values against magnitude lists for both JSON files on the same plot
    plt.figure(figsize=(10, 8))
    for i in range(len(all_data)):
        file_name = json_files[i]
        label = "CMOS" if "rms_mags_phot_NG1109-2807_1.json" in file_name else "CCD"
        plt.plot(additional_data[tmag]['mag'], additional_data[tmag]['rms'], 'o', label=label)
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
