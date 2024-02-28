#!/usr/bin/env python

import datetime
import json
import os
import fnmatch
import argparse
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


def plot_images():
    # Set plot parameters
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.minor.left'] = True

    # Font and fontsize
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14

    # Legend
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 14


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


def plot_lc(table, gaia_id_to_plot, bin_size=1):
    # Select rows with the specified Gaia ID
    gaia_id_data = table[table['gaia_id'] == gaia_id_to_plot]

    # Get jd_mid, flux_2, and sky_2 for the selected rows
    jd_mid = gaia_id_data['jd_mid']
    flux_2 = gaia_id_data['flux_2']
    fluxerr_2 = gaia_id_data['fluxerr_2']
    flux_w_sky_2 = gaia_id_data['flux_w_sky_2']
    fluxerr_w_sky_2 = gaia_id_data['fluxerr_w_sky_2']
    sky_2 = flux_w_sky_2 - flux_2
    skyerr_2 = np.sqrt(fluxerr_2**2 + fluxerr_w_sky_2**2)

    # Bin the data
    jd_mid_binned = [np.mean(jd_mid[i:i+bin_size]) for i in range(0, len(jd_mid), bin_size)]
    flux_2_binned = [np.mean(flux_2[i:i+bin_size]) for i in range(0, len(flux_2), bin_size)]
    fluxerr_2_binned = [np.sqrt(np.sum(fluxerr_2[i:i+bin_size]**2)) / bin_size for i in range(0, len(fluxerr_2), bin_size)]
    sky_2_binned = [np.mean(sky_2[i:i+bin_size]) for i in range(0, len(sky_2), bin_size)]
    skyerr_2_binned = [np.sqrt(np.sum(skyerr_2[i:i+bin_size]**2)) / bin_size for i in range(0, len(skyerr_2), bin_size)]

    # Plot jd_mid vs flux_2
    plt.errorbar(jd_mid_binned, flux_2_binned, yerr=fluxerr_2_binned, fmt='o', color='black', label='Flux 2')
    plt.errorbar(jd_mid_binned, sky_2_binned, yerr=skyerr_2_binned, fmt='o', color='blue', label='Sky 2')

    # Add labels and title
    plt.xlabel('MJD [days]')
    plt.ylabel('Flux [e-]')
    plt.title(f'LC for Gaia ID {gaia_id_to_plot}')
    plt.legend()
    plt.show()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific Gaia ID')
    parser.add_argument('gaia_id', type=int, help='The Gaia ID of the star to plot')
    parser.add_argument('--bin', type=int, default=1, help='Number of images to bin')
    args = parser.parse_args()
    gaia_id_to_plot = args.gaia_id
    bin_size = args.bin

    # Set plot parameters
    plot_images()

    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    # Plot the first photometry file
    print(f"Plotting the first photometry file {phot_files[0]}...")
    phot_table = read_phot_file(phot_files[0])
    plot_lc(phot_table, gaia_id_to_plot, bin_size)


if __name__ == "__main__":
    main()