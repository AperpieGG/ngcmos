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


def bin_time_flux_error(time, flux, error, bin_fact):
    """
    Use reshape to bin light curve data, clip under filled bins
    Works with 2D arrays of flux and errors

    Note: under filled bins are clipped off the end of the series

    Parameters
    ----------
    time : array         of times to bin
    flux : array         of flux values to bin
    error : array         of error values to bin
    bin_fact : int
        Number of measurements to combine

    Returns
    -------
    times_b : array
        Binned times
    flux_b : array
        Binned fluxes
    error_b : array
        Binned errors

    Raises
    ------
    None
    """
    n_binned = int(len(time) / bin_fact)
    clip = n_binned * bin_fact
    time_b = np.average(time[:clip].reshape(n_binned, bin_fact), axis=1)
    # determine if 1 or 2d flux/err inputs
    if len(flux.shape) == 1:
        flux_b = np.average(flux[:clip].reshape(n_binned, bin_fact), axis=1)
        error_b = np.sqrt(np.sum(error[:clip].reshape(n_binned, bin_fact) ** 2, axis=1)) / bin_fact
    else:
        # assumed 2d with 1 row per star
        n_stars = len(flux)
        flux_b = np.average(flux[:clip].reshape((n_stars, n_binned, bin_fact)), axis=2)
        error_b = np.sqrt(np.sum(error[:clip].reshape((n_stars, n_binned, bin_fact)) ** 2, axis=2)) / bin_fact
    return time_b, flux_b, error_b


def plot_lc(table, gaia_id_to_plot, bin_size=1, exposure_time=10):
    # Select rows with the specified Gaia ID
    gaia_id_data = table[table['gaia_id'] == gaia_id_to_plot]
    tmag = gaia_id_data['Tmag'][0]

    # Get jd_mid, flux_2, and sky_2 for the selected rows
    jd_mid = gaia_id_data['jd_mid']
    fluxes = [gaia_id_data[f'flux_{i}'] for i in range(2, 7)]
    fluxerrs = [gaia_id_data[f'fluxerr_{i}'] for i in range(2, 7)]
    sky = [gaia_id_data[f'flux_w_sky_{i}'] - gaia_id_data[f'flux_{i}'] for i in range(2, 7)]
    skyerrs = [np.sqrt(gaia_id_data[f'fluxerr_{i}'] ** 2 + gaia_id_data[f'fluxerr_w_sky_{i}'] ** 2) for i in range(2, 7)]
    x = gaia_id_data['x'][0]
    y = gaia_id_data['y'][0]

    for i in range(5):
        # Bin flux data
        jd_mid_binned, fluxes[i], fluxerrs[i] = bin_time_flux_error(jd_mid, fluxes[i], fluxerrs[i], bin_size)
        # Bin sky data using the same binned jd_mid as the flux data
        _, sky[i], skyerrs[i] = bin_time_flux_error(jd_mid, sky[i], skyerrs[i], bin_size)

    # Determine the bin label for the y-axis
    bin_label = f'binned {bin_size * exposure_time / 60:.2f} min'

    fig, axs = plt.subplots(3, 2, figsize=(16, 10))

    # Plot jd_mid vs flux_2
    for i in range(5):
        row = i // 2
        col = i % 2
        axs[row, col].errorbar(jd_mid, fluxes[i], yerr=fluxerrs[i], fmt='o', color='black', label=f'Flux {i+2}')
        axs[row, col].errorbar(jd_mid, sky[i], yerr=skyerrs[i], fmt='o', color='blue', label=f'Sky {i+2}')
        axs[row, col].set_xlabel('MJD [days]')
        axs[row, col].set_ylabel(f'Flux [e-] {bin_label}')
        axs[row, col].legend()

    # Plot x, y positions
    print(len(jd_mid), len(x), len(y))
    axs[2, 1].plot(jd_mid, x, 'o', color='red', label='X Position')
    axs[2, 1].plot(jd_mid, y, 'o', color='green', label='Y Position')
    axs[2, 1].set_xlabel('MJD [days]')
    axs[2, 1].set_ylabel('Position')
    axs[2, 1].legend()

    fig.suptitle(f'LC for Gaia ID {gaia_id_to_plot} (Tmag = {tmag:.2f} mag), on position {x:.2f}, {y:.2f}')
    plt.tight_layout()
    plt.show()


def plot_lc_for_all_stars(table, bin_size):
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
