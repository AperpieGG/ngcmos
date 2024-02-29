#!/usr/bin/env python

import datetime
import json
import os
import fnmatch
from datetime import datetime, timedelta
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from utils import plot_images
from wotan import flatten


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


def plot_lc_detrend_all_stars(table):
    # Get unique Gaia IDs from the table
    unique_gaia_ids = np.unique(table['gaia_id'])

    # Iterate over each unique Gaia ID
    for gaia_id_to_plot in unique_gaia_ids:
        # Select rows with the specified Gaia ID
        gaia_id_data = table[table['gaia_id'] == gaia_id_to_plot]

        # Check if Tmag is available for the selected Gaia ID
        if 'Tmag' in gaia_id_data.columns:
            tmag = gaia_id_data['Tmag'].iloc[0]  # Get Tmag for the first row of the selected Gaia ID

            # Plot light curve and detrended light curve
            plot_detrended_lc(gaia_id_data, gaia_id_to_plot, tmag)


def plot_detrended_lc(table, gaia_id_to_plot):
    # Select rows with the specified Gaia ID
    gaia_id_data = table[table['gaia_id'] == gaia_id_to_plot]

    # Get jd_mid, flux_2, and fluxerr_2 for the selected rows
    jd_mid = gaia_id_data['jd_mid']
    flux_2 = gaia_id_data['flux_2']
    fluxerr_2 = gaia_id_data['fluxerr_2']
    tmag = gaia_id_data['Tmag']

    # Use wotan to detrend the light curve
    detrended_flux, trend = flatten(jd_mid, flux_2, method='mean', window_length=0.05, return_trend=True)

    relative_flux = flux_2 / trend
    relative_err = fluxerr_2 / trend

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot raw flux with wotan model
    ax1.plot(jd_mid, flux_2, 'o', color='black', label='Raw Flux 2')
    ax1.plot(jd_mid, trend, color='red', label='Wotan Model')
    ax1.set_xlabel('MJD [days]')
    ax1.set_ylabel('Flux [e-]')
    ax1.legend()

    # Plot detrended flux
    ax2.errorbar(jd_mid, relative_flux, yerr=relative_err, fmt='o', color='black', label='Detrended Flux')
    ax2.set_ylabel('Detrended Flux [e-]')
    ax2.set_title(f'Detrended LC for Gaia ID {gaia_id_to_plot} (Tmag = {tmag:.2f})')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    # Set plot parameters
    plot_images()

    # Get the current night directory
    current_night_directory = find_current_night_directory(base_path)

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    # Plot each photometry file
    for phot_file in phot_files:
        print(f"Plotting photometry file {phot_file}...")
        phot_table = read_phot_file(phot_file)

        # Plot detrended light curves for all stars in the photometry file
        plot_lc_detrend_all_stars(phot_table)

        plt.show()


if __name__ == "__main__":
    main()


