#!/usr/bin/env python

import datetime
import json
import os
import fnmatch
import argparse
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


def plot_lc(table, gaia_id_to_plot, bin_size=1, exposure_time=10):
    # Select rows with the specified Gaia ID
    gaia_id_data = table[table['gaia_id'] == gaia_id_to_plot]
    tmag = gaia_id_data['Tmag'][0]

    # Get jd_mid, flux_2, and sky_2 for the selected rows
    jd_mid = gaia_id_data['jd_mid']
    flux_2 = gaia_id_data['flux_2']
    fluxerr_2 = gaia_id_data['fluxerr_2']
    flux_3 = gaia_id_data['flux_3']
    fluxerr_3 = gaia_id_data['fluxerr_3']
    flux_4 = gaia_id_data['flux_4']
    fluxerr_4 = gaia_id_data['fluxerr_4']
    flux_5 = gaia_id_data['flux_5']
    fluxerr_5 = gaia_id_data['fluxerr_5']
    flux_6 = gaia_id_data['flux_6']
    fluxerr_6 = gaia_id_data['fluxerr_6']
    flux_w_sky_2 = gaia_id_data['flux_w_sky_2']
    fluxerr_w_sky_2 = gaia_id_data['fluxerr_w_sky_2']
    flux_w_sky_3 = gaia_id_data['flux_w_sky_3']
    fluxerr_w_sky_3 = gaia_id_data['fluxerr_w_sky_3']
    flux_w_sky_4 = gaia_id_data['flux_w_sky_4']
    fluxerr_w_sky_4 = gaia_id_data['fluxerr_w_sky_4']
    flux_w_sky_5 = gaia_id_data['flux_w_sky_5']
    fluxerr_w_sky_5 = gaia_id_data['fluxerr_w_sky_5']
    flux_w_sky_6 = gaia_id_data['flux_w_sky_6']
    fluxerr_w_sky_6 = gaia_id_data['fluxerr_w_sky_6']
    sky_2 = flux_w_sky_2 - flux_2
    skyerr_2 = np.sqrt(fluxerr_2 ** 2 + fluxerr_w_sky_2 ** 2)
    sky_3 = flux_w_sky_3 - flux_3
    skyerr_3 = np.sqrt(fluxerr_3 ** 2 + fluxerr_w_sky_3 ** 2)
    sky_4 = flux_w_sky_4 - flux_4
    skyerr_4 = np.sqrt(fluxerr_4 ** 2 + fluxerr_w_sky_4 ** 2)
    sky_5 = flux_w_sky_5 - flux_5
    skyerr_5 = np.sqrt(fluxerr_5 ** 2 + fluxerr_w_sky_5 ** 2)
    sky_6 = flux_w_sky_6 - flux_6
    skyerr_6 = np.sqrt(fluxerr_6 ** 2 + fluxerr_w_sky_6 ** 2)

    # Bin the data
    jd_mid_binned = [np.mean(jd_mid[i:i + bin_size]) for i in range(0, len(jd_mid), bin_size)]
    flux_2_binned = [np.mean(flux_2[i:i + bin_size]) for i in range(0, len(flux_2), bin_size)]
    fluxerr_2_binned = [np.sqrt(np.sum(fluxerr_2[i:i + bin_size] ** 2)) / bin_size for i in
                        range(0, len(fluxerr_2), bin_size)]
    flux_3_binned = [np.mean(flux_3[i:i + bin_size]) for i in range(0, len(flux_3), bin_size)]
    fluxerr_3_binned = [np.sqrt(np.sum(fluxerr_3[i:i + bin_size] ** 2)) / bin_size for i in
                        range(0, len(fluxerr_3), bin_size)]
    flux_4_binned = [np.mean(flux_4[i:i + bin_size]) for i in range(0, len(flux_4), bin_size)]
    fluxerr_4_binned = [np.sqrt(np.sum(fluxerr_4[i:i + bin_size] ** 2)) / bin_size for i in
                        range(0, len(fluxerr_4), bin_size)]
    flux_5_binned = [np.mean(flux_5[i:i + bin_size]) for i in range(0, len(flux_5), bin_size)]
    fluxerr_5_binned = [np.sqrt(np.sum(fluxerr_5[i:i + bin_size] ** 2)) / bin_size for i in
                        range(0, len(fluxerr_5), bin_size)]
    flux_6_binned = [np.mean(flux_6[i:i + bin_size]) for i in range(0, len(flux_6), bin_size)]
    fluxerr_6_binned = [np.sqrt(np.sum(fluxerr_6[i:i + bin_size] ** 2)) / bin_size for i in
                        range(0, len(fluxerr_6), bin_size)]
    sky_2_binned = [np.mean(sky_2[i:i + bin_size]) for i in range(0, len(sky_2), bin_size)]
    skyerr_2_binned = [np.sqrt(np.sum(skyerr_2[i:i + bin_size] ** 2)) / bin_size for i in
                       range(0, len(skyerr_2), bin_size)]
    sky_3_binned = [np.mean(sky_3[i:i + bin_size]) for i in range(0, len(sky_3), bin_size)]
    skyerr_3_binned = [np.sqrt(np.sum(skyerr_3[i:i + bin_size] ** 2)) / bin_size for i in
                       range(0, len(skyerr_3), bin_size)]
    sky_4_binned = [np.mean(sky_4[i:i + bin_size]) for i in range(0, len(sky_4), bin_size)]
    skyerr_4_binned = [np.sqrt(np.sum(skyerr_4[i:i + bin_size] ** 2)) / bin_size for i in
                       range(0, len(skyerr_4), bin_size)]
    sky_5_binned = [np.mean(sky_5[i:i + bin_size]) for i in range(0, len(sky_5), bin_size)]
    skyerr_5_binned = [np.sqrt(np.sum(skyerr_5[i:i + bin_size] ** 2)) / bin_size for i in
                       range(0, len(skyerr_5), bin_size)]
    sky_6_binned = [np.mean(sky_6[i:i + bin_size]) for i in range(0, len(sky_6), bin_size)]
    skyerr_6_binned = [np.sqrt(np.sum(skyerr_6[i:i + bin_size] ** 2)) / bin_size for i in
                       range(0, len(skyerr_6), bin_size)]

    # Plot jd_mid vs flux_2
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.errorbar(jd_mid_binned, flux_2_binned, yerr=fluxerr_2_binned, fmt='o', color='black', label='Flux 2')
    plt.errorbar(jd_mid_binned, sky_2_binned, yerr=skyerr_2_binned, fmt='o', color='blue', label='Sky 2')
    plt.xlabel('MJD [days]')
    plt.ylabel(f'Flux [e-] {bin_size * exposure_time / 60:.2f} min')
    plt.title(f'LC for Gaia ID {gaia_id_to_plot} (Tmag = {tmag:.2f} mag)')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.errorbar(jd_mid_binned, flux_3_binned, yerr=fluxerr_3_binned, fmt='o', color='red', label='Flux 3')
    plt.errorbar(jd_mid_binned, sky_3_binned, yerr=skyerr_3_binned, fmt='o', color='green', label='Sky 3')
    plt.xlabel('MJD [days]')
    plt.ylabel(f'Flux [e-] {bin_size * exposure_time / 60:.2f} min')
    plt.title(f'LC for Gaia ID {gaia_id_to_plot} (Tmag = {tmag:.2f} mag)')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.errorbar(jd_mid_binned, flux_4_binned, yerr=fluxerr_4_binned, fmt='o', color='orange', label='Flux 4')
    plt.errorbar(jd_mid_binned, sky_4_binned, yerr=skyerr_4_binned, fmt='o', color='purple', label='Sky 4')
    plt.xlabel('MJD [days]')
    plt.ylabel(f'Flux [e-] {bin_size * exposure_time / 60:.2f} min')
    plt.title(f'LC for Gaia ID {gaia_id_to_plot} (Tmag = {tmag:.2f} mag)')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.errorbar(jd_mid_binned, flux_5_binned, yerr=fluxerr_5_binned, fmt='o', color='magenta', label='Flux 5')
    plt.errorbar(jd_mid_binned, sky_5_binned, yerr=skyerr_5_binned, fmt='o', color='cyan', label='Sky 5')
    plt.xlabel('MJD [days]')
    plt.ylabel(f'Flux [e-] {bin_size * exposure_time / 60:.2f} min')
    plt.title(f'LC for Gaia ID {gaia_id_to_plot} (Tmag = {tmag:.2f} mag)')
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.errorbar(jd_mid_binned, flux_6_binned, yerr=fluxerr_6_binned, fmt='o', color='brown', label='Flux 6')
    plt.errorbar(jd_mid_binned, sky_6_binned, yerr=skyerr_6_binned, fmt='o', color='gray', label='Sky 6')
    plt.xlabel('MJD [days]')
    plt.ylabel(f'Flux [e-] {bin_size * exposure_time / 60:.2f} min')
    plt.title(f'LC for Gaia ID {gaia_id_to_plot} (Tmag = {tmag:.2f} mag)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_lc_for_all_stars(table, bin_size=1):
    # Get unique Gaia IDs from the table
    unique_gaia_ids = np.unique(table['gaia_id'])

    # Iterate over each unique Gaia ID
    for gaia_id_to_plot in unique_gaia_ids:
        # Plot the light curve for the current Gaia ID
        plot_lc(table, gaia_id_to_plot, bin_size)


def plot_lc_with_detrend(table, gaia_id_to_plot):
    # Select rows with the specified Gaia ID
    gaia_id_data = table[table['gaia_id'] == gaia_id_to_plot]

    # Get jd_mid, flux_2, and fluxerr_2 for the selected rows
    jd_mid = gaia_id_data['jd_mid']
    flux_2 = gaia_id_data['flux_2']
    fluxerr_2 = gaia_id_data['fluxerr_2']
    tmag = gaia_id_data['Tmag'][0]

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot light curve for a specific Gaia ID')
    parser.add_argument('--gaia_id', type=int, help='The Gaia ID of the star to plot')
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

    if gaia_id_to_plot is None:
        plot_lc_for_all_stars(phot_table, bin_size)
    else:
        plot_lc_with_detrend(phot_table, gaia_id_to_plot)

    plt.show()


if __name__ == "__main__":
    main()
