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


def plot_lc(table):
    # Get the first gaia_id from the first row
    first_gaia_id = table['gaia_id'][750]

    # Select rows with the same gaia_id as the first one
    gaia_id_data = table[table['gaia_id'] == first_gaia_id]

    # Get jd_mid and flux_2 for the selected rows
    jd_mid = gaia_id_data['jd_mid']
    flux_2 = gaia_id_data['flux_2']
    fluxerr_2 = gaia_id_data['fluxerr_2']
    flux_w_sky_2 = gaia_id_data['flux_w_sky_2']
    fluxerr_w_sky_2 = gaia_id_data['fluxerr_w_sky_2']
    sky_2 = fluxerr_w_sky_2 - flux_2

    # Plot jd_mid vs flux_2
    plt.errorbar(jd_mid, flux_2, yerr=fluxerr_2, fmt='o', color='black', label='Flux 2')
    plt.errorbar(jd_mid, sky_2, yerr=fluxerr_w_sky_2, fmt='o', color='red', label='Sky bgk 2')

    # Add labels and title
    plt.xlabel('MJD [days]')
    plt.ylabel('Flux 2 [e-]')
    plt.title(f'LC for Gaia ID {first_gaia_id}')
    plt.legend()
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
    plot_lc(phot_table)


if __name__ == "__main__":
    main()
