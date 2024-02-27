#! /usr/bin/env python
import datetime
import json
import os
import fnmatch
from datetime import datetime, timedelta
import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt


def load_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config


# Load paths from the configuration file
config = load_config('directories.json')
calibration_paths = config["calibration_paths"]
base_paths = config["base_paths"]
out_paths = config["out_paths"]

# Select directory based on existence
for calibration_path, base_path, out_path in zip(calibration_paths, base_paths, out_paths):
    if os.path.exists(base_path):
        break


def find_current_night_directory(directory):
    """
    Find the directory for the current night based on the current date.
    If not found, use the current working directory.

    Parameters
    ----------
    directory : str
        Base path for the directory.

    Returns
    -------
    str
        Path to the current night directory.
    """
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    current_date_directory = os.path.join(directory, previous_date)
    return current_date_directory if os.path.isdir(current_date_directory) else os.getcwd()


def get_phot_files(directory):
    """
    Get photometry files with the pattern 'phot_*.fits' from the directory.

    Parameters
    ----------
    directory : str
        Directory containing the files.

    Returns
    -------
    list of str
        List of photometry files matching the pattern.
    """
    phot_files = []
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, 'phot_*.fits'):
            phot_files.append(os.path.join(directory, filename))
    return phot_files


def read_phot_file(filename):
    """
    Read the photometry file.

    Parameters
    ----------
    filename : str
        Photometry file to read.

    Returns
    -------
    astropy.table.table.Table
        Table containing the photometry data.
    """
    # Read the photometry file here using fits or any other appropriate method
    try:
        with fits.open(filename) as ff:
            # Access the data in the photometry file as needed
            tab = ff[1].data
            return tab
    except Exception as e:
        print(f"Error reading photometry file {filename}: {e}")
        return None


def plot_first_gaia_id_vs_jd_mid(table):
    # Select data for the first image
    first_image_data = table[table['frame_id'] == table['frame_id'][0]]

    # Get unique gaia_ids in the first image
    unique_gaia_ids = set(first_image_data['gaia_id'])

    # Iterate over each unique gaia_id
    for gaia_id in unique_gaia_ids:
        # Select rows corresponding to the current gaia_id
        mask = first_image_data['gaia_id'] == gaia_id
        gaia_id_data = first_image_data[mask]

        # Get the first jd_mid and flux_2 for the current gaia_id
        first_jd_mid = gaia_id_data['jd_mid'][0]
        first_flux_2 = gaia_id_data['flux_2'][0]

        # Plot jd_mid vs flux_2 for the current gaia_id
        plt.scatter(first_jd_mid, first_flux_2, label=gaia_id)

    # Add labels and legend
    plt.xlabel('JD Mid')
    plt.ylabel('Flux 2')
    plt.title('First Image: Flux 2 vs JD Mid for Each Gaia ID')
    plt.show()


def main():
    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    # Plot the first photometry file
    print(f"Plotting the first photometry file {phot_files[0]}...")
    phot_table = read_phot_file(phot_files[0])
    plot_first_gaia_id_vs_jd_mid(phot_table)


if __name__ == "__main__":
    main()
